// INT8 processing element for 2x2 systolic array
// direct weight load
module pe (
    input  wire       clk,
    input  wire       rst_n,

    // north (partial sum input)
    input  wire [15:0] pe_psum_in,

    // weight load (directly from FSM)
    input  wire [7:0]  pe_weight_in,
    input  wire        pe_load_w,

    // west (activation + control)
    input  wire [7:0]  pe_input_in,
    input  wire        pe_valid_in,

    // south (partial sum output)
    output reg  [15:0] pe_psum_out,

    // east (activation + control out)
    output reg  [7:0]  pe_input_out,
    output reg         pe_valid_out
);

    reg [7:0] weight;

    // INT8 MAC: signed 8x8 -> 16-bit product + 16-bit psum
    wire signed [15:0] product;
    assign product = $signed(pe_input_in) * $signed(weight);

    wire signed [15:0] mac_result;
    assign mac_result = product + $signed(pe_psum_in);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_input_out <= 8'd0;
            pe_psum_out  <= 16'd0;
            pe_valid_out <= 1'b0;
        end
        else begin
            pe_valid_out <= pe_valid_in;

            if (pe_valid_in) begin
                pe_input_out <= pe_input_in;
                pe_psum_out  <= mac_result[15:0];
            end
        end
    end

    always @(posedge clk) begin
        if (pe_load_w)
            weight <= pe_weight_in;
    end

endmodule