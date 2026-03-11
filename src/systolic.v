// 2x2 INT8 systolic array
// direct weight loading
module systolic (
    input  wire clk,
    input  wire rst_n,

    // left side: 8-bit activation inputs
    input  wire [7:0] sys_data_in_11,
    input  wire [7:0] sys_data_in_21,

    input wire sys_start_1,    // valid signal for row 1
    input wire sys_start_2,    // valid signal for row 2

    // bottom: 16-bit accumulated outputs
    output wire [15:0] sys_data_out_21,
    output wire [15:0] sys_data_out_22,
    output wire        sys_valid_out_21,
    output wire        sys_valid_out_22,

    // direct weight inputs for each PE
    input  wire [7:0] sys_weight_in_11,
    input  wire [7:0] sys_weight_in_12,
    input  wire [7:0] sys_weight_in_21,
    input  wire [7:0] sys_weight_in_22,
    input  wire       sys_load_w        // load weight pulse (all PEs)
);

    // inter-PE wires: activation (8-bit, left→right)
    wire [7:0] pe_input_out_11;
    wire [7:0] pe_input_out_21;

    // inter-PE wires: psum (16-bit, top→bottom)
    wire [15:0] pe_psum_out_11;
    wire [15:0] pe_psum_out_12;

    // inter-PE wires: control
    wire pe_valid_out_11;
    wire pe_valid_out_21;
    wire pe_valid_out_22;

    // ROW 1

    pe pe11 (
        .clk(clk), .rst_n(rst_n),
        .pe_valid_in(sys_start_1),     .pe_valid_out(pe_valid_out_11),
        .pe_load_w(sys_load_w),
        .pe_input_in(sys_data_in_11),  .pe_input_out(pe_input_out_11),
        .pe_psum_in(16'd0),            .pe_psum_out(pe_psum_out_11),
        .pe_weight_in(sys_weight_in_11)
    );

    wire _unused_valid_12;
    pe pe12 (
        .clk(clk), .rst_n(rst_n),
        .pe_valid_in(pe_valid_out_11), .pe_valid_out(_unused_valid_12),
        .pe_load_w(sys_load_w),
        .pe_input_in(pe_input_out_11), .pe_input_out(),
        .pe_psum_in(16'd0),            .pe_psum_out(pe_psum_out_12),
        .pe_weight_in(sys_weight_in_12)
    );

    // ROW 2

    pe pe21 (
        .clk(clk), .rst_n(rst_n),
        .pe_valid_in(sys_start_2),     .pe_valid_out(pe_valid_out_21),
        .pe_load_w(sys_load_w),
        .pe_input_in(sys_data_in_21),  .pe_input_out(pe_input_out_21),
        .pe_psum_in(pe_psum_out_11),   .pe_psum_out(sys_data_out_21),
        .pe_weight_in(sys_weight_in_21)
    );

    pe pe22 (
        .clk(clk), .rst_n(rst_n),
        .pe_valid_in(pe_valid_out_21), .pe_valid_out(pe_valid_out_22),
        .pe_load_w(sys_load_w),
        .pe_input_in(pe_input_out_21), .pe_input_out(),
        .pe_psum_in(pe_psum_out_12),   .pe_psum_out(sys_data_out_22),
        .pe_weight_in(sys_weight_in_22)
    );

    assign sys_valid_out_21 = pe_valid_out_21;
    assign sys_valid_out_22 = pe_valid_out_22;

    wire _unused_sys = &{_unused_valid_12, 1'b0};

endmodule