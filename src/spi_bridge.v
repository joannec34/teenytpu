// SPI slave bridge
// 16-bit result shift-out
//
// protocol:
//   1. host asserts CS_N low
//   2. host clocks 8-bit opcode on MOSI (MSB first)
//   3. opcode-dependent address/data follows
//   4. host deasserts CS_N
//
// ppcodes:
//   0x01 WRITE_WEIGHT : 1-byte col_sel + 2x 8-bit weights (top -> bottom)
//   0x02 LOAD_INPUT   : 1-byte row_sel + 1x 8-bit activation
//   0x03 CMD_START    : trigger computation (no payload)
//   0x04 READ_RESULT  : 1-byte col_sel, device returns 1x 16-bit psum (MSB first)
//   0x05 READ_STATUS  : device returns 1 status byte {6'b0, done, busy}
//
module spi_bridge (
    input  wire       clk,
    input  wire       rst_n,

    // SPI pins
    input  wire       spi_sclk,
    input  wire       spi_cs_n,
    input  wire       spi_mosi,
    output reg        spi_miso,

    // weight load interface
    output reg        wt_valid,
    output reg  [7:0] wt_data,
    output reg        wt_col_sel,  // 0 = col0, 1 = col1
    output reg        wt_row_sel,  // 0 = row0, 1 = row1

    // activation load interface
    output reg        act_valid,
    output reg  [7:0] act_data,
    output reg        act_row_sel, // 0 = row0, 1 = row1

    // control
    output reg        cmd_start,

    // result readback (single 16-bit value)
    output reg        res_req,
    output reg        res_col_sel,
    input  wire [15:0] res_data,

    // status
    input  wire       sts_busy,
    input  wire       sts_done
);

    // CDC synchronizers (2-FF)
    reg [2:0] sclk_sync;
    reg [1:0] cs_sync;
    reg [1:0] mosi_sync;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync <= 3'd0;
            cs_sync   <= 2'b11;
            mosi_sync <= 2'd0;
        end
        else begin
            sclk_sync <= {sclk_sync[1:0], spi_sclk};
            cs_sync   <= {cs_sync[0], spi_cs_n};
            mosi_sync <= {mosi_sync[0], spi_mosi};
        end
    end

    wire cs_active = ~cs_sync[1];
    wire sclk_rise = (sclk_sync[2:1] == 2'b01);
    wire sclk_fall = (sclk_sync[2:1] == 2'b10);
    wire mosi_bit  = mosi_sync[1];

    // FSM
    localparam [3:0]
        S_IDLE       = 4'd0,
        S_OPCODE     = 4'd1,
        S_WT_COL     = 4'd2,
        S_WT_DATA    = 4'd3,
        S_ACT_ROW    = 4'd4,
        S_ACT_DATA   = 4'd5,
        S_RD_COL     = 4'd6,
        S_RD_SHIFT   = 4'd7,
        S_RD_STATUS  = 4'd8,
        S_RD_LATCH   = 4'd9;

    reg [3:0] state;
    reg [7:0] shift_in;
    reg [2:0] bit_cnt;
    reg [15:0] shift_out;  // 16-bit for result or 8-bit for status
    reg [4:0]  out_bits;   // max 15 for 16-bit shift
    reg        wt_phase;   // 0 = first weight byte, 1 = second

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            shift_in   <= 8'd0;
            bit_cnt    <= 3'd0;
            shift_out  <= 16'd0;
            out_bits   <= 5'd0;
            spi_miso   <= 1'b0;
            wt_valid   <= 1'b0;
            wt_data    <= 8'd0;
            wt_col_sel <= 1'b0;
            wt_row_sel <= 1'b0;
            wt_phase   <= 1'b0;
            act_valid  <= 1'b0;
            act_data   <= 8'd0;
            act_row_sel<= 1'b0;
            cmd_start  <= 1'b0;
            res_req    <= 1'b0;
            res_col_sel<= 1'b0;
        end
        else begin
            // default pulse-clears
            wt_valid  <= 1'b0;
            act_valid <= 1'b0;
            cmd_start <= 1'b0;
            res_req   <= 1'b0;

            // CS deassert -> reset FSM
            if (!cs_active) begin
                state   <= S_IDLE;
                bit_cnt <= 3'd0;
            end
            else begin
                // MISO driven on falling SCLK edge
                if (sclk_fall && (state == S_RD_SHIFT || state == S_RD_STATUS)) begin
                    spi_miso  <= shift_out[15];
                    shift_out <= {shift_out[14:0], 1'b0};
                    if (out_bits != 5'd0)
                        out_bits <= out_bits - 5'd1;
                end

                // S_RD_LATCH: one system-clock delay (NOT gated by sclk_rise)
                if (state == S_RD_LATCH) begin
                    shift_out <= res_data;
                    out_bits  <= 5'd15;
                    state     <= S_RD_SHIFT;
                end

                // sample MOSI on rising SCLK edge
                if (sclk_rise) begin
                    case (state)

                        S_IDLE: begin
                            state   <= S_OPCODE;
                            bit_cnt <= 3'd6;
                            shift_in <= {7'd0, mosi_bit};
                        end

                        S_OPCODE: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                bit_cnt <= 3'd7;
                                case ({shift_in[6:0], mosi_bit})
                                    8'h01: state <= S_WT_COL;
                                    8'h02: state <= S_ACT_ROW;
                                    8'h03: begin
                                        cmd_start <= 1'b1;
                                        state <= S_IDLE;
                                    end
                                    8'h04: state <= S_RD_COL;
                                    8'h05: begin
                                        state <= S_RD_STATUS;
                                        shift_out <= {6'b0, sts_done, sts_busy, 8'd0};
                                        out_bits  <= 5'd7;
                                    end
                                    default: state <= S_IDLE;
                                endcase
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        // WRITE_WEIGHT
                        S_WT_COL: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                wt_col_sel <= mosi_bit;
                                wt_phase   <= 1'b0;
                                bit_cnt    <= 3'd7;
                                state      <= S_WT_DATA;
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        S_WT_DATA: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                wt_data    <= {shift_in[6:0], mosi_bit};
                                wt_row_sel <= wt_phase;
                                wt_valid   <= 1'b1;
                                if (wt_phase) begin
                                    state <= S_IDLE;
                                end
                                else begin
                                    wt_phase <= 1'b1;
                                    bit_cnt  <= 3'd7;
                                end
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        // LOAD_INPUT
                        S_ACT_ROW: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                act_row_sel <= mosi_bit;
                                bit_cnt     <= 3'd7;
                                state       <= S_ACT_DATA;
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        S_ACT_DATA: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                act_data  <= {shift_in[6:0], mosi_bit};
                                act_valid <= 1'b1;
                                state     <= S_IDLE;
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        // READ_RESULT
                        S_RD_COL: begin
                            shift_in <= {shift_in[6:0], mosi_bit};
                            if (bit_cnt == 3'd0) begin
                                res_col_sel <= mosi_bit;
                                res_req     <= 1'b1;
                                state       <= S_RD_LATCH;
                            end
                            else
                                bit_cnt <= bit_cnt - 3'd1;
                        end

                        S_RD_SHIFT: begin
                            // bits shifted out on sclk_fall above
                        end

                        S_RD_STATUS: begin
                            // bits shifted out on sclk_fall above
                        end

                        default: state <= S_IDLE;

                    endcase
                end // sclk_rise
            end // cs_active
        end // !rst_n
    end

    wire _unused_spi = &{shift_in[7], 1'b0};

endmodule
