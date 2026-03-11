<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

TeenyTPU is a 2x2 INT8 systolic array TPU designed. At its core, it features a 2x2 grid of Processing Elements (PEs) that can perform matrix multiplication operations on INT8 data, resulting in 16-bit partial sum sequences.

**SPI Pin Mapping**

- **`ui[0]`**: SPI SCLK
- **`ui[1]`**: SPI CS_N
- **`ui[2]`**: SPI MOSI
- **`uo[0]`**: SPI MISO
- **`uo[1]`**: BUSY flag
- **`uo[2]`**: DONE flag

**Architecture Overview**

- **SPI Slave Bridge** — deserialises SPI transactions from an external host into register writes and command pulses. Uses 2-FF CDC synchronisers for SCLK, CS_N, and MOSI. Supports five opcodes:
  - `0x01 WRITE_WEIGHT` — load an 8-bit weight into a selected column/row.
  - `0x02 LOAD_INPUT` — load an 8-bit activation into a selected row.
  - `0x03 CMD_START` — trigger the compute FSM.
  - `0x04 READ_RESULT` — read back two 16-bit partial sums from a selected column.
  - `0x05 READ_STATUS` — read the `{done, busy}` status byte.
- **Control FSM** — sequences weight loading, shadow→active switch, activation feeding, draining, and result readback through six states (`IDLE → LOAD_W → SWITCH → FEED → DRAIN → DONE`).
- **2×2 Systolic Array** — four PE instances connected in a grid. Activations flow left -> right, partial sums flow top -> bottom, and weights propagate top→bottom during loading.

## How to test

To test the TeenyTPU, you must implement an SPI master to drive the designated `ui` pins.

1. **Hardware Reset**: Provide a clock signal (`clk`) and assert `rst_n` low briefly to reset the FSM and the systolic array.
2. **Load Weights**: Assert `CS_N` low, send the `0x01` opcode, followed by the column index (e.g., `0x00`), and then the two 8-bit weights for that column. Deassert `CS_N`. Repeat this for column 1.
3. **Load Activations**: Assert `CS_N` low, send the `0x02` opcode, followed by the row index (e.g., `0x00`), and the 8-bit activation value. Deassert `CS_N`. Repeat this for row 1.
4. **Trigger Computation**: Send the `0x03` opcode to initiate computation. The `busy` pin (`uo[1]`) will assert high.
5. **Poll Status**: Wait for the `done` pin (`uo[2]`) to assert high, which indicates that the systolic array has finished computation. Alternatively, you can use the `0x05` opcode to read the `{6'b0, done, busy}` status byte repeatedly.
6. **Read Results**: Send the `0x04` opcode followed by the column index (`0x00` or `0x01`). The TPU will respond by shifting out 16 bits of result data (`MISO`) representing the partial sums of that column.

## External hardware

An SPI master connected to `ui[0]` (SCLK), `ui[1]` (CS_N), `ui[2]` (MOSI), and `uo[0]` (MISO).
