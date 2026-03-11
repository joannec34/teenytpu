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
    reg [7:0]  w_reg [0:1][0:1]; // [col][row]

    // activation storage (2 rows)
    reg [7:0]  a_reg [0:1];      // [row]

    // systolic control signals
    reg        sys_load_w;
    reg        sys_start_1, sys_start_2;
    reg [7:0]  sys_data_in_11, sys_data_in_21;

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
        CTL_LOAD_W   = 3'd1,   // load all weights into PEs (single cycle)
        CTL_FEED1    = 3'd2,   // feed row-1 activations
        CTL_FEED2    = 3'd3,   // feed row-2 activations (staggered)
        CTL_DRAIN    = 3'd4,   // wait for valid outputs
        CTL_DONE     = 3'd5;

    reg [2:0] ctl_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctl_state      <= CTL_IDLE;
            ctl_busy       <= 1'b0;
            ctl_done       <= 1'b0;
            sys_load_w     <= 1'b0;
            sys_start_1    <= 1'b0;
            sys_start_2    <= 1'b0;
            sys_data_in_11 <= 8'd0;
            sys_data_in_21 <= 8'd0;
        end
        else begin
            // default: deassert single-cycle pulses
            sys_load_w  <= 1'b0;
            sys_start_1 <= 1'b0;
            sys_start_2 <= 1'b0;

            // control FSM
            case (ctl_state)

                CTL_IDLE: begin
                    if (cmd_start) begin
                        ctl_done  <= 1'b0;
                        ctl_busy  <= 1'b1;
                        ctl_state <= CTL_LOAD_W;
                    end
                end

                // single cycle: load all 4 weights directly into PEs
                CTL_LOAD_W: begin
                    sys_load_w <= 1'b1;
                    ctl_state  <= CTL_FEED1;
                end

                // feed row-1 activation
                CTL_FEED1: begin
                    sys_start_1    <= 1'b1;
                    sys_data_in_11 <= a_reg[0];
                    ctl_state      <= CTL_FEED2;
                end

                // feed row-2 activation (staggered by 1 cycle)
                CTL_FEED2: begin
                    sys_start_2    <= 1'b1;
                    sys_data_in_21 <= a_reg[1];
                    ctl_state      <= CTL_DRAIN;
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

    // SPI write handlers (purely synchronous, no reset)
    always @(posedge clk) begin
        if (wt_valid)
            w_reg[wt_col_sel][wt_row_sel] <= wt_data;

        if (act_valid)
            a_reg[act_row_sel] <= act_data;
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
        .sys_data_in_11(sys_data_in_11),
        .sys_data_in_21(sys_data_in_21),
        .sys_start_1(sys_start_1),
        .sys_start_2(sys_start_2),
        .sys_data_out_21(sys_out_21),
        .sys_data_out_22(sys_out_22),
        .sys_valid_out_21(sys_valid_21),
        .sys_valid_out_22(sys_valid_22),
        .sys_weight_in_11(w_reg[0][0]),
        .sys_weight_in_12(w_reg[1][0]),
        .sys_weight_in_21(w_reg[0][1]),
        .sys_weight_in_22(w_reg[1][1]),
        .sys_load_w(sys_load_w)
    );

endmodule