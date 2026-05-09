// top-level: connects SPI bridge <-> control FSM <-> 2x2 systolic array
// single-cycle weight load, no double-buffering
module tpu (
    input  wire       clk,
    input  wire       rst_n,

    // SPI pins (directly from Tiny Tapeout IO)
    input  wire       spi_sclk,
    input  wire       spi_cs_n,
    input  wire       spi_mosi,
    output wire       spi_miso,

    // status
    output wire       busy,
    output wire       done
);

    // SPI bridge <-> control wires
    wire        wt_valid, wt_col_sel, wt_row_sel;
    wire [7:0]  wt_data;
    wire        act_valid, act_row_sel;
    wire [7:0]  act_data;
    wire        cmd_start;
    wire        res_req, res_col_sel;
    wire [15:0] res_data_0;
    wire        _unused_tpu = &{res_req, 1'b0};

    // control FSM state
    reg        ctl_busy;
    reg        ctl_done;

    // weight storage (2 cols x 2 rows = 4 weights)
    reg [7:0] w00, w01, w10, w11; // w{col}{row}

    // activation storage (2 rows)
    reg [7:0] a0, a1;

    // systolic control signals
    reg        sys_start_1, sys_start_2;

    // result mux
    wire [15:0] sys_out_21, sys_out_22;
    wire        sys_valid_21, sys_valid_22;

    assign busy = ctl_busy;
    assign done = ctl_done;

    // result readback mux
    assign res_data_0 = res_col_sel ? sys_out_22 : sys_out_21;

    // FSM states
    localparam [2:0]
        CTL_IDLE     = 3'd0,
        CTL_FEED1    = 3'd1,   // feed row-1 activations
        CTL_FEED2    = 3'd2,   // feed row-2 activations (staggered)
        CTL_DRAIN    = 3'd3,   // wait for valid outputs
        CTL_DONE     = 3'd4;

    reg [2:0] ctl_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctl_state      <= CTL_IDLE;
            ctl_busy       <= 1'b0;
            ctl_done       <= 1'b0;
            sys_start_1    <= 1'b0;
            sys_start_2    <= 1'b0;
            w00 <= 8'd0; w01 <= 8'd0;
            w10 <= 8'd0; w11 <= 8'd0;
            a0  <= 8'd0; a1  <= 8'd0;
        end
        else begin
            // default: deassert single-cycle pulses
            sys_start_1 <= 1'b0;
            sys_start_2 <= 1'b0;

            // SPI write handlers (active anytime)
            if (wt_valid) begin
                case ({wt_col_sel, wt_row_sel})
                    2'b00: w00 <= wt_data;
                    2'b01: w01 <= wt_data;
                    2'b10: w10 <= wt_data;
                    2'b11: w11 <= wt_data;
                endcase
            end

            if (act_valid) begin
                if (act_row_sel) a1 <= act_data;
                else             a0 <= act_data;
            end

            // control FSM
            case (ctl_state)

                CTL_IDLE: begin
                    if (cmd_start) begin
                        ctl_done  <= 1'b0;
                        ctl_busy  <= 1'b1;
                        ctl_state <= CTL_FEED1;
                    end
                end

                // feed row-1 activation
                CTL_FEED1: begin
                    sys_start_1 <= 1'b1;
                    ctl_state   <= CTL_FEED2;
                end

                // feed row-2 activation (staggered by 1 cycle)
                CTL_FEED2: begin
                    sys_start_2 <= 1'b1;
                    ctl_state   <= CTL_DRAIN;
                end

                // wait for valid outputs from bottom row PEs
                CTL_DRAIN: begin
                    if (sys_valid_21 || sys_valid_22) begin
                        ctl_state <= CTL_DONE;
                    end
                end

                CTL_DONE: begin
                    ctl_busy  <= 1'b0;
                    ctl_done  <= 1'b1;
                    ctl_state <= CTL_IDLE;
                end

                default: ctl_state <= CTL_IDLE;

            endcase
        end
    end

    // SPI bridge instance
    spi_bridge u_spi (
        .clk(clk), .rst_n(rst_n),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n),
        .spi_mosi(spi_mosi), .spi_miso(spi_miso),
        .wt_valid(wt_valid),   .wt_data(wt_data),
        .wt_col_sel(wt_col_sel), .wt_row_sel(wt_row_sel),
        .act_valid(act_valid), .act_data(act_data),
        .act_row_sel(act_row_sel),
        .cmd_start(cmd_start),
        .res_req(res_req),     .res_col_sel(res_col_sel),
        .res_data(res_data_0),
        .sts_busy(ctl_busy),   .sts_done(ctl_done)
    );

    // systolic array instance
    systolic u_sys (
        .clk(clk), .rst_n(rst_n),
        .sys_data_in_11(a0),
        .sys_data_in_21(a1),
        .sys_start_1(sys_start_1),
        .sys_start_2(sys_start_2),
        .sys_data_out_21(sys_out_21),
        .sys_data_out_22(sys_out_22),
        .sys_valid_out_21(sys_valid_21),
        .sys_valid_out_22(sys_valid_22),
        .sys_weight_in_11(w00),
        .sys_weight_in_12(w10),
        .sys_weight_in_21(w01),
        .sys_weight_in_22(w11)
    );

endmodule