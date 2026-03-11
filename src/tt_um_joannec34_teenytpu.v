/*
 * Copyright (c) 2024 Your Name
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_joannec34_teenytpu (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  // All output pins must be assigned. If not used, assign to 0.

    // PIN MAPPING
    // ui_in[0] = SPI SCLK
    // ui_in[1] = SPI CS_N
    // ui_in[2] = SPI MOSI
    // uo_out[0] = SPI MISO
    // uo_out[1] = busy
    // uo_out[2] = done

    wire spi_miso;
    wire tpu_busy;
    wire tpu_done;

    tpu u_tpu (
        .clk(clk),
        .rst_n(rst_n),
        .spi_sclk(ui_in[0]),
        .spi_cs_n(ui_in[1]),
        .spi_mosi(ui_in[2]),
        .spi_miso(spi_miso),
        .busy(tpu_busy),
        .done(tpu_done)
    );

    assign uo_out[0] = spi_miso;
    assign uo_out[1] = tpu_busy;
    assign uo_out[2] = tpu_done;
    assign uo_out[7:3] = 5'd0;

    assign uio_out = 8'd0;
    assign uio_oe  = 8'd0;

  // List all unused inputs to prevent warnings
    wire _unused = &{ena, ui_in[7:3], uio_in, 1'b0};

endmodule
