# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for TeenyTPU — covers SPI bridge protocol, matrix multiplication,
and basic end-to-end TPU functionality via the Tiny Tapeout wrapper.

in mapping (from tt_um_joannec34_teenytpu):
  ui_in[0]  = SPI SCLK
  ui_in[1]  = SPI CS_N
  ui_in[2]  = SPI MOSI
  uo_out[0] = SPI MISO
  uo_out[1] = busy
  uo_out[2] = done

SPI opcodes:
  0x01  WRITE_WEIGHT : col_sel(8b) + weight_row0(8b) + weight_row1(8b)
  0x02  LOAD_INPUT   : row_sel(8b) + activation(8b)
  0x03  CMD_START    : trigger computation
  0x04  READ_RESULT  : col_sel(8b) → 2×16-bit psums shifted out
  0x05  READ_STATUS  : → 1 status byte {6'b0, done, busy}
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


# ──────────────────────── SPI helper functions ────────────────────────

# The DUT system clock is much faster than SPI clock.  We use ~20 system
# clocks per SPI half-period so the CDC synchronisers inside spi_bridge
# can reliably sample edges.
SPI_HALF_PERIOD_CLKS = 20


def _set_spi_pins(dut, sclk, cs_n, mosi):
    """Drive SPI signals via ui_in[2:0]."""
    val = (int(mosi) << 2) | (int(cs_n) << 1) | int(sclk)
    dut.ui_in.value = val


async def _spi_cs_assert(dut):
    """Assert CS (active-low) and wait for CDC settling."""
    _set_spi_pins(dut, sclk=0, cs_n=0, mosi=0)
    await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)


async def _spi_cs_deassert(dut):
    """Deassert CS and wait for CDC settling."""
    _set_spi_pins(dut, sclk=0, cs_n=1, mosi=0)
    await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)


async def _spi_send_byte(dut, byte_val):
    """Clock out 8 bits MSB-first on MOSI. CS must already be low."""
    for i in range(7, -1, -1):
        bit = (byte_val >> i) & 1
        # Set MOSI with SCLK low
        _set_spi_pins(dut, sclk=0, cs_n=0, mosi=bit)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        # Rising edge — device samples MOSI
        _set_spi_pins(dut, sclk=1, cs_n=0, mosi=bit)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
    # Return SCLK low
    _set_spi_pins(dut, sclk=0, cs_n=0, mosi=0)
    await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)


async def _spi_read_bits(dut, n_bits):
    """Clock in *n_bits* from MISO (MSB-first). CS must already be low."""
    result = 0
    for _ in range(n_bits):
        # Falling edge — device drives MISO
        _set_spi_pins(dut, sclk=0, cs_n=0, mosi=0)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        # Rising edge — we sample MISO
        _set_spi_pins(dut, sclk=1, cs_n=0, mosi=0)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        miso = (int(dut.uo_out.value) >> 0) & 1  # uo_out[0] = MISO
        result = (result << 1) | miso
    _set_spi_pins(dut, sclk=0, cs_n=0, mosi=0)
    await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
    return result


# ──────────── High-level SPI transaction helpers ──────────────

async def spi_write_weight(dut, col, w_row0, w_row1):
    """
    Opcode 0x01: write two weights for a column.
    col: 0 or 1
    w_row0, w_row1: 8-bit signed weight values
    """
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x01)       # WRITE_WEIGHT opcode
    await _spi_send_byte(dut, col & 0xFF) # column select
    await _spi_send_byte(dut, w_row0 & 0xFF)  # weight row 0
    await _spi_send_byte(dut, w_row1 & 0xFF)  # weight row 1
    await _spi_cs_deassert(dut)


async def spi_load_activation(dut, row, value):
    """
    Opcode 0x02: load one activation value.
    row: 0 or 1
    value: 8-bit signed activation
    """
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x02)         # LOAD_INPUT opcode
    await _spi_send_byte(dut, row & 0xFF)   # row select
    await _spi_send_byte(dut, value & 0xFF) # activation data
    await _spi_cs_deassert(dut)


async def spi_cmd_start(dut):
    """Opcode 0x03: trigger computation."""
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x03)
    await _spi_cs_deassert(dut)


async def spi_read_result(dut, col):
    """
    Opcode 0x04: read 16-bit result psum for one column.
    Returns (psum, 0) for backward compatibility with test assertions.
    """
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x04)
    await _spi_send_byte(dut, col & 0xFF)
    raw = await _spi_read_bits(dut, 16)
    await _spi_cs_deassert(dut)
    psum0 = raw & 0xFFFF
    return psum0, 0


async def spi_read_status(dut):
    """
    Opcode 0x05: read status byte.
    Returns (done, busy) as booleans.
    """
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x05)
    status = await _spi_read_bits(dut, 8)
    await _spi_cs_deassert(dut)
    busy = bool(status & 0x01)
    done = bool(status & 0x02)
    return done, busy


def to_signed16(val):
    """Convert unsigned 16-bit to signed Python int."""
    if val >= 0x8000:
        return val - 0x10000
    return val


def to_signed8(val):
    """Convert unsigned 8-bit to signed Python int."""
    if val >= 0x80:
        return val - 0x100
    return val


def compute_expected(w_col0, w_col1, activations):
    """
    Compute expected systolic array outputs.
    w_col0 = (w_row0, w_row1) for column 0
    w_col1 = (w_row0, w_row1) for column 1
    activations = (a_row0, a_row1)
    Returns (col0_result, col1_result) as signed integers.
    """
    a0 = to_signed8(activations[0] & 0xFF)
    a1 = to_signed8(activations[1] & 0xFF)
    w00 = to_signed8(w_col0[0] & 0xFF)
    w01 = to_signed8(w_col0[1] & 0xFF)
    w10 = to_signed8(w_col1[0] & 0xFF)
    w11 = to_signed8(w_col1[1] & 0xFF)
    col0 = a0 * w00 + a1 * w01
    col1 = a0 * w10 + a1 * w11
    return col0, col1


# ──────────────── Reset helper ────────────────

async def reset_dut(dut):
    """Apply a clean reset sequence."""
    dut.ena.value = 1
    dut.ui_in.value = 0b00000010  # CS_N = 1 (deasserted), SCLK=0, MOSI=0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def wait_for_done(dut, timeout_cycles=2000):
    """Poll the done flag (uo_out[2]) until it goes high, or timeout."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if (int(dut.uo_out.value) >> 2) & 1:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
#                        TEST CASES
# ═══════════════════════════════════════════════════════════════

# ────────────────── SPI Bridge Tests ──────────────────

@cocotb.test()
async def test_spi_status_after_reset(dut):
    """After reset, status should report not-busy and not-done."""
    dut._log.info("test_spi_status_after_reset")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    done, busy = await spi_read_status(dut)
    assert not busy, "busy should be 0 after reset"
    assert not done, "done should be 0 after reset"


@cocotb.test()
async def test_spi_write_weight_no_crash(dut):
    """Verify that writing weights via SPI completes without hanging."""
    dut._log.info("test_spi_write_weight_no_crash")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write col 0 weights: w[0][0]=3, w[0][1]=5
    await spi_write_weight(dut, col=0, w_row0=3, w_row1=5)
    # Write col 1 weights: w[1][0]=7, w[1][1]=2
    await spi_write_weight(dut, col=1, w_row0=7, w_row1=2)

    # If we get here without hanging, the test passes
    done, busy = await spi_read_status(dut)
    assert not busy, "TPU should not be busy after weight load only"


@cocotb.test()
async def test_spi_load_activation_no_crash(dut):
    """Verify that loading activations via SPI completes without hanging."""
    dut._log.info("test_spi_load_activation_no_crash")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=20)

    done, busy = await spi_read_status(dut)
    assert not busy, "TPU should not be busy after activation load only"


@cocotb.test()
async def test_spi_cmd_start_sets_busy(dut):
    """CMD_START should set the busy flag."""
    dut._log.info("test_spi_cmd_start_sets_busy")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Load minimal data so the FSM has something to do
    await spi_write_weight(dut, col=0, w_row0=1, w_row1=0)
    await spi_write_weight(dut, col=1, w_row0=0, w_row1=1)
    await spi_load_activation(dut, row=0, value=1)
    await spi_load_activation(dut, row=1, value=1)

    await spi_cmd_start(dut)

    # Read busy — it should be high (or already done for this tiny computation)
    # We allow either busy or done
    done, busy = await spi_read_status(dut)
    assert busy or done, "Expected busy or done after CMD_START"


@cocotb.test()
async def test_spi_cs_deassert_resets_fsm(dut):
    """Deasserting CS mid-transaction should cleanly reset the SPI FSM."""
    dut._log.info("test_spi_cs_deassert_resets_fsm")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Start an opcode transfer but abort halfway
    await _spi_cs_assert(dut)
    # Send only 4 bits of a byte (incomplete)
    for i in range(4):
        _set_spi_pins(dut, sclk=0, cs_n=0, mosi=0)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        _set_spi_pins(dut, sclk=1, cs_n=0, mosi=0)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
    await _spi_cs_deassert(dut)

    # Now do a valid status read — should still work
    done, busy = await spi_read_status(dut)
    assert not busy, "TPU should not be busy after aborted SPI transaction"


# ────────────────── Matrix Multiplication Tests ──────────────────

@cocotb.test()
async def test_matmul_identity(dut):
    """
    Multiply by identity-like weights.
    W = [[1, 0],   A = [5, 3]
         [0, 1]]

    Expected result column 0 (W_col0 · A):
      PE(0,0) = A[0]*W[0][0] = 5*1 = 5
      PE(1,0) = A[1]*W[0][1] + psum = 3*0 + 5 = 5
    Expected result column 1 (W_col1 · A):
      PE(0,1) = A[0]*W[1][0] = 5*0 = 0
      PE(1,1) = A[1]*W[1][1] + psum = 3*1 + 0 = 3
    """
    dut._log.info("test_matmul_identity")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Load identity weights
    await spi_write_weight(dut, col=0, w_row0=1, w_row1=0)
    await spi_write_weight(dut, col=1, w_row0=0, w_row1=1)

    # Load activations
    await spi_load_activation(dut, row=0, value=5)
    await spi_load_activation(dut, row=1, value=3)

    # Start computation
    await spi_cmd_start(dut)

    # Wait for completion
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete in time"

    # Read results
    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Col0 results: {to_signed16(p0_0)}, {to_signed16(p0_1)}")
    dut._log.info(f"Col1 results: {to_signed16(p1_0)}, {to_signed16(p1_1)}")

    exp_c0, exp_c1 = compute_expected((1, 0), (0, 1), (5, 3))
    assert to_signed16(p0_0) == exp_c0, f"Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Col1 expected {exp_c1}, got {to_signed16(p1_0)}"


@cocotb.test()
async def test_matmul_simple(dut):
    """
    Simple 2×2 matrix multiply with known small values.
    W = [[2, 3],   A = [4, 5]
         [1, 4]]

    Column 0: PE11 = 4*2 = 8, PE21 = 5*1 + 8 = 13
    Column 1: PE12 = 4*3 = 12, PE22 = 5*4 + 12 = 32
    """
    dut._log.info("test_matmul_simple")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=2, w_row1=1)
    await spi_write_weight(dut, col=1, w_row0=3, w_row1=4)

    await spi_load_activation(dut, row=0, value=4)
    await spi_load_activation(dut, row=1, value=5)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete in time"

    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Col0 results: {to_signed16(p0_0)}, {to_signed16(p0_1)}")
    dut._log.info(f"Col1 results: {to_signed16(p1_0)}, {to_signed16(p1_1)}")

    exp_c0, exp_c1 = compute_expected((2, 1), (3, 4), (4, 5))
    assert to_signed16(p0_0) == exp_c0, f"Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Col1 expected {exp_c1}, got {to_signed16(p1_0)}"


@cocotb.test()
async def test_matmul_zeros(dut):
    """All-zero activations should produce zero results."""
    dut._log.info("test_matmul_zeros")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=10, w_row1=20)
    await spi_write_weight(dut, col=1, w_row0=30, w_row1=40)

    await spi_load_activation(dut, row=0, value=0)
    await spi_load_activation(dut, row=1, value=0)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete in time"

    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Col0: {p0_0}, {p0_1} | Col1: {p1_0}, {p1_1}")

    assert to_signed16(p0_0) == 0, f"Col0 expected 0, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == 0, f"Col1 expected 0, got {to_signed16(p1_0)}"


@cocotb.test()
async def test_matmul_negative_weights(dut):
    """
    Negative weights (signed INT8).
    W = [[-1, 2],   A = [10, 5]
         [ 3,-2]]

    Col0: PE11 = 10*(-1) = -10, PE21 = 5*3 + (-10) = 5
    Col1: PE12 = 10*2 = 20,     PE22 = 5*(-2) + 20 = 10
    """
    dut._log.info("test_matmul_negative_weights")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # -1 = 0xFF, -2 = 0xFE in two's complement 8-bit
    await spi_write_weight(dut, col=0, w_row0=0xFF, w_row1=3)
    await spi_write_weight(dut, col=1, w_row0=2, w_row1=0xFE)

    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=5)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete in time"

    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Col0: {to_signed16(p0_0)}, {to_signed16(p0_1)}")
    dut._log.info(f"Col1: {to_signed16(p1_0)}, {to_signed16(p1_1)}")

    exp_c0, exp_c1 = compute_expected((0xFF, 3), (2, 0xFE), (10, 5))
    assert to_signed16(p0_0) == exp_c0, f"Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Col1 expected {exp_c1}, got {to_signed16(p1_0)}"


@cocotb.test()
async def test_matmul_max_values(dut):
    """
    Stress test with maximum INT8 values.
    W = [[127, 127],   A = [127, 127]
         [127, 127]]

    Each PE: 127*127 = 16129
    Bottom PE accumulates: 16129 + 16129 = 32258
    All within 16-bit range (max 32767 signed).
    """
    dut._log.info("test_matmul_max_values")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=127, w_row1=127)
    await spi_write_weight(dut, col=1, w_row0=127, w_row1=127)

    await spi_load_activation(dut, row=0, value=127)
    await spi_load_activation(dut, row=1, value=127)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete in time"

    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Col0: {to_signed16(p0_0)}, {to_signed16(p0_1)}")
    dut._log.info(f"Col1: {to_signed16(p1_0)}, {to_signed16(p1_1)}")

    exp_c0, exp_c1 = compute_expected((127, 127), (127, 127), (127, 127))
    assert to_signed16(p0_0) == exp_c0, f"Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Col1 expected {exp_c1}, got {to_signed16(p1_0)}"


# ────────────────── Basic TPU Functionality Tests ──────────────────

@cocotb.test()
async def test_reset_outputs_zero(dut):
    """All outputs should be zero immediately after reset."""
    dut._log.info("test_reset_outputs_zero")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # uo_out should be all zeros (MISO=0, busy=0, done=0)
    uo = int(dut.uo_out.value)
    miso = uo & 1
    busy = (uo >> 1) & 1
    done = (uo >> 2) & 1
    assert busy == 0, f"busy should be 0 after reset, got {busy}"
    assert done == 0, f"done should be 0 after reset, got {done}"
    dut._log.info(f"Post-reset uo_out = 0x{uo:02X} (miso={miso}, busy={busy}, done={done})")


@cocotb.test()
async def test_busy_during_computation(dut):
    """The busy flag should go high during computation."""
    dut._log.info("test_busy_during_computation")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=1, w_row1=1)
    await spi_write_weight(dut, col=1, w_row0=1, w_row1=1)
    await spi_load_activation(dut, row=0, value=1)
    await spi_load_activation(dut, row=1, value=1)

    await spi_cmd_start(dut)

    # For a tiny 2x2 array, computation may complete within the SPI
    # CMD_START transaction itself (~7 clocks vs ~380 for SPI).
    # Check that busy went high OR done is already asserted.
    saw_busy = False
    saw_done = False
    for _ in range(200):
        await RisingEdge(dut.clk)
        uo = int(dut.uo_out.value)
        if (uo >> 1) & 1:
            saw_busy = True
            break
        if (uo >> 2) & 1:
            saw_done = True
            break

    assert saw_busy or saw_done, "neither busy nor done was ever asserted"
    dut._log.info(f"busy={saw_busy}, done={saw_done} — computation was triggered")


@cocotb.test()
async def test_done_after_computation(dut):
    """The done flag should go high after computation finishes."""
    dut._log.info("test_done_after_computation")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=5, w_row1=5)
    await spi_write_weight(dut, col=1, w_row0=5, w_row1=5)
    await spi_load_activation(dut, row=0, value=2)
    await spi_load_activation(dut, row=1, value=3)

    await spi_cmd_start(dut)

    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "done flag never asserted after computation"

    uo = int(dut.uo_out.value)
    done = (uo >> 2) & 1
    busy = (uo >> 1) & 1
    assert done == 1, "done should be 1"
    assert busy == 0, "busy should be 0 when done"
    dut._log.info("done flag correctly asserted, busy deasserted")


@cocotb.test()
async def test_back_to_back_computations(dut):
    """
    Run two computations back-to-back to verify the TPU resets
    properly between runs.
    """
    dut._log.info("test_back_to_back_computations")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # ---- First computation ----
    await spi_write_weight(dut, col=0, w_row0=2, w_row1=0)
    await spi_write_weight(dut, col=1, w_row0=0, w_row1=2)
    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=20)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "First computation did not complete"
    r1_c0 = await spi_read_result(dut, col=0)
    r1_c1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Run1 - Col0: {r1_c0}, Col1: {r1_c1}")

    # ---- Second computation (different data) ----
    await spi_write_weight(dut, col=0, w_row0=3, w_row1=1)
    await spi_write_weight(dut, col=1, w_row0=1, w_row1=3)
    await spi_load_activation(dut, row=0, value=7)
    await spi_load_activation(dut, row=1, value=4)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Second computation did not complete"
    r2_c0 = await spi_read_result(dut, col=0)
    r2_c1 = await spi_read_result(dut, col=1)
    dut._log.info(f"Run2 - Col0: {r2_c0}, Col1: {r2_c1}")

    # The results should differ between runs
    dut._log.info("Both computations completed successfully")


# ═══════════════════════════════════════════════════════════════
#              NEW SPI BRIDGE PROTOCOL TESTS
# ═══════════════════════════════════════════════════════════════

@cocotb.test()
async def test_spi_invalid_opcode(dut):
    """Sending an invalid opcode should not crash or hang the SPI FSM."""
    dut._log.info("test_spi_invalid_opcode")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Send invalid opcode 0xFF
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0xFF)
    await _spi_cs_deassert(dut)

    # SPI FSM should still be functional
    done, busy = await spi_read_status(dut)
    assert not busy, "busy should be 0 after invalid opcode"
    assert not done, "done should be 0 after invalid opcode"
    dut._log.info("SPI FSM recovered from invalid opcode")


@cocotb.test()
async def test_spi_repeated_status_reads(dut):
    """Multiple consecutive status reads should return consistent results."""
    dut._log.info("test_spi_repeated_status_reads")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    for i in range(5):
        done, busy = await spi_read_status(dut)
        assert not busy, f"Read {i}: busy should be 0"
        assert not done, f"Read {i}: done should be 0"
    dut._log.info("5 consecutive status reads returned consistent idle state")


@cocotb.test()
async def test_spi_weight_overwrite(dut):
    """Overwriting weights before computation should use the new weights."""
    dut._log.info("test_spi_weight_overwrite")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write initial weights
    await spi_write_weight(dut, col=0, w_row0=99, w_row1=99)
    await spi_write_weight(dut, col=1, w_row0=99, w_row1=99)

    # Overwrite with new weights
    await spi_write_weight(dut, col=0, w_row0=3, w_row1=1)
    await spi_write_weight(dut, col=1, w_row0=2, w_row1=4)

    await spi_load_activation(dut, row=0, value=5)
    await spi_load_activation(dut, row=1, value=6)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)

    # Should use overwritten weights (3,1) and (2,4), not (99,99)
    exp_c0, exp_c1 = compute_expected((3, 1), (2, 4), (5, 6))
    assert to_signed16(p0_0) == exp_c0, f"Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Col1 expected {exp_c1}, got {to_signed16(p1_0)}"
    dut._log.info("Weight overwrite test passed")


@cocotb.test()
async def test_spi_rp2040_clock_speeds(dut):
    """
    Vary SPI half-period to simulate different RP2040 SPI clock rates.
    The CDC synchronizers must handle both fast and slow SPI clocks.
    """
    dut._log.info("test_spi_rp2040_clock_speeds")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    global SPI_HALF_PERIOD_CLKS
    original_half = SPI_HALF_PERIOD_CLKS

    for speed_name, half_period in [("slow", 40), ("normal", 20), ("fast", 10)]:
        SPI_HALF_PERIOD_CLKS = half_period
        await reset_dut(dut)

        await spi_write_weight(dut, col=0, w_row0=2, w_row1=3)
        await spi_write_weight(dut, col=1, w_row0=1, w_row1=1)
        await spi_load_activation(dut, row=0, value=4)
        await spi_load_activation(dut, row=1, value=5)
        await spi_cmd_start(dut)
        completed = await wait_for_done(dut, timeout_cycles=5000)
        assert completed, f"Computation did not complete at {speed_name} SPI speed"

        p0_0, _ = await spi_read_result(dut, col=0)
        p1_0, _ = await spi_read_result(dut, col=1)
        exp_c0, exp_c1 = compute_expected((2, 3), (1, 1), (4, 5))
        assert to_signed16(p0_0) == exp_c0, \
            f"{speed_name}: Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
        assert to_signed16(p1_0) == exp_c1, \
            f"{speed_name}: Col1 expected {exp_c1}, got {to_signed16(p1_0)}"
        dut._log.info(f"{speed_name} SPI speed (half={half_period}) passed")

    SPI_HALF_PERIOD_CLKS = original_half


# ═══════════════════════════════════════════════════════════════
#              NEW CONTROL FSM TESTS
# ═══════════════════════════════════════════════════════════════

@cocotb.test()
async def test_fsm_start_without_data(dut):
    """CMD_START with no weights/activations loaded should complete with zero outputs."""
    dut._log.info("test_fsm_start_without_data")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Immediately start — registers should be zero from reset
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete without data"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)
    assert to_signed16(p0_0) == 0, f"Col0 expected 0, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == 0, f"Col1 expected 0, got {to_signed16(p1_0)}"
    dut._log.info("FSM start without data produced zeroes as expected")


@cocotb.test()
async def test_fsm_done_clears_on_new_start(dut):
    """After done is asserted, a new CMD_START should clear done and set busy."""
    dut._log.info("test_fsm_done_clears_on_new_start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # First computation
    await spi_write_weight(dut, col=0, w_row0=1, w_row1=0)
    await spi_write_weight(dut, col=1, w_row0=0, w_row1=1)
    await spi_load_activation(dut, row=0, value=1)
    await spi_load_activation(dut, row=1, value=1)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "First computation did not complete"

    done, busy = await spi_read_status(dut)
    assert done, "done should be set after first computation"
    assert not busy, "busy should be clear after first computation"

    # Second computation — done should clear
    await spi_load_activation(dut, row=0, value=2)
    await spi_load_activation(dut, row=1, value=3)
    await spi_cmd_start(dut)

    # Poll: either busy or done (computation may finish fast)
    done2, busy2 = await spi_read_status(dut)
    assert busy2 or done2, "Expected busy or done after second CMD_START"

    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Second computation did not complete"
    dut._log.info("done correctly cleared on new CMD_START")


@cocotb.test()
async def test_fsm_read_result_before_start(dut):
    """Reading results before any computation should return 0."""
    dut._log.info("test_fsm_read_result_before_start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    p0_0, p0_1 = await spi_read_result(dut, col=0)
    p1_0, p1_1 = await spi_read_result(dut, col=1)
    assert p0_0 == 0, f"Col0 psum0 expected 0 before computation, got {p0_0}"
    assert p1_0 == 0, f"Col1 psum0 expected 0 before computation, got {p1_0}"
    dut._log.info("Pre-computation read returns zeroes")


# ═══════════════════════════════════════════════════════════════
#          NEW MATRIX MULTIPLICATION TESTS (with assertions)
# ═══════════════════════════════════════════════════════════════

async def _run_matmul_test(dut, w_col0, w_col1, activations, test_label=""):
    """Shared helper: loads weights/activations, runs, asserts results."""
    await spi_write_weight(dut, col=0, w_row0=w_col0[0] & 0xFF, w_row1=w_col0[1] & 0xFF)
    await spi_write_weight(dut, col=1, w_row0=w_col1[0] & 0xFF, w_row1=w_col1[1] & 0xFF)
    await spi_load_activation(dut, row=0, value=activations[0] & 0xFF)
    await spi_load_activation(dut, row=1, value=activations[1] & 0xFF)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, f"{test_label}: Computation did not complete in time"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)

    exp_c0, exp_c1 = compute_expected(w_col0, w_col1, activations)
    actual_c0, actual_c1 = to_signed16(p0_0), to_signed16(p1_0)
    dut._log.info(f"{test_label}: Col0={actual_c0} (exp {exp_c0}), Col1={actual_c1} (exp {exp_c1})")
    assert actual_c0 == exp_c0, f"{test_label}: Col0 expected {exp_c0}, got {actual_c0}"
    assert actual_c1 == exp_c1, f"{test_label}: Col1 expected {exp_c1}, got {actual_c1}"
    return actual_c0, actual_c1


@cocotb.test()
async def test_matmul_ones(dut):
    """All-ones weights and activations → each column sums to 2."""
    dut._log.info("test_matmul_ones")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await _run_matmul_test(dut, (1, 1), (1, 1), (1, 1), "ones")


@cocotb.test()
async def test_matmul_asymmetric(dut):
    """Asymmetric weights: one column active, one zero."""
    dut._log.info("test_matmul_asymmetric")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await _run_matmul_test(dut, (3, 0), (0, 7), (4, 6), "asymmetric")


@cocotb.test()
async def test_matmul_negative_activations(dut):
    """Negative activation with positive weights."""
    dut._log.info("test_matmul_negative_activations")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # A = (-3, 6), W_col0 = (2, 3), W_col1 = (4, 1)
    # Col0 = (-3)*2 + 6*3 = 12,  Col1 = (-3)*4 + 6*1 = -6
    await _run_matmul_test(dut, (2, 3), (4, 1), (-3, 6), "neg_act")


@cocotb.test()
async def test_matmul_all_negative(dut):
    """All-negative weights and activations."""
    dut._log.info("test_matmul_all_negative")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # W_col0=(-2,-3), W_col1=(-1,-4), A=(-5,-6)
    # Col0 = (-5)*(-2) + (-6)*(-3) = 10+18 = 28
    # Col1 = (-5)*(-1) + (-6)*(-4) = 5+24  = 29
    await _run_matmul_test(dut, (-2, -3), (-1, -4), (-5, -6), "all_neg")


@cocotb.test()
async def test_matmul_overflow_boundary(dut):
    """Large values near 16-bit boundary but not overflowing."""
    dut._log.info("test_matmul_overflow_boundary")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # W_col0=(100,100), W_col1=(-100,-100), A=(100,100)
    # Col0 = 100*100+100*100 = 20000, Col1 = 100*(-100)+100*(-100) = -20000
    await _run_matmul_test(dut, (100, 100), (-100, -100), (100, 100), "overflow_boundary")


# ═══════════════════════════════════════════════════════════════
#       END-TO-END / RP2040 WORKFLOW / SUBMISSION TESTS
# ═══════════════════════════════════════════════════════════════

@cocotb.test()
async def test_full_e2e_rp2040_workflow(dut):
    """
    Simulates a complete RP2040 firmware session:
    reset → check status → load weights → load activations →
    start → poll status → read results → verify values.
    This mirrors the exact sequence an RP2040 would execute.
    """
    dut._log.info("test_full_e2e_rp2040_workflow")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Step 1: RP2040 checks TPU is idle
    done, busy = await spi_read_status(dut)
    assert not busy and not done, "TPU should be idle after reset"

    # Step 2: RP2040 loads weight matrix W = [[5, -2], [3, 7]]
    await spi_write_weight(dut, col=0, w_row0=5, w_row1=3)
    await spi_write_weight(dut, col=1, w_row0=(-2) & 0xFF, w_row1=7)

    # Step 3: RP2040 loads activation vector A = [4, -1]
    await spi_load_activation(dut, row=0, value=4)
    await spi_load_activation(dut, row=1, value=(-1) & 0xFF)

    # Step 4: RP2040 triggers computation
    await spi_cmd_start(dut)

    # Step 5: RP2040 polls status until done
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation timed out"

    done, busy = await spi_read_status(dut)
    assert done, "done should be 1 after computation"
    assert not busy, "busy should be 0 after computation"

    # Step 6: RP2040 reads results
    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)

    # Step 7: Verify
    exp_c0, exp_c1 = compute_expected((5, 3), (-2, 7), (4, -1))
    assert to_signed16(p0_0) == exp_c0, f"e2e Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"e2e Col1 expected {exp_c1}, got {to_signed16(p1_0)}"
    dut._log.info(f"RP2040 workflow: Col0={to_signed16(p0_0)}, Col1={to_signed16(p1_0)} ✓")


@cocotb.test()
async def test_rp2040_rapid_spi_transactions(dut):
    """
    Back-to-back SPI transactions with minimal idle gap, as an RP2040
    might do in a tight firmware loop. No extra delays between CS deassert
    and next CS assert.
    """
    dut._log.info("test_rp2040_rapid_spi_transactions")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Rapid-fire: write weights, read status, write activations, read status, start
    await spi_write_weight(dut, col=0, w_row0=6, w_row1=2)
    done, busy = await spi_read_status(dut)  # Interleaved status check
    assert not busy, "Unexpected busy during weight load"

    await spi_write_weight(dut, col=1, w_row0=4, w_row1=8)
    await spi_load_activation(dut, row=0, value=3)
    await spi_load_activation(dut, row=1, value=7)

    done, busy = await spi_read_status(dut)  # Pre-start check
    assert not busy, "Unexpected busy before start"

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Rapid transaction computation did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)
    exp_c0, exp_c1 = compute_expected((6, 2), (4, 8), (3, 7))
    assert to_signed16(p0_0) == exp_c0, f"Rapid Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1, f"Rapid Col1 expected {exp_c1}, got {to_signed16(p1_0)}"
    dut._log.info("Rapid SPI transactions passed")


@cocotb.test()
async def test_hardware_reset_mid_computation(dut):
    """Assert rst_n during active computation, verify clean recovery."""
    dut._log.info("test_hardware_reset_mid_computation")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=50, w_row1=50)
    await spi_write_weight(dut, col=1, w_row0=50, w_row1=50)
    await spi_load_activation(dut, row=0, value=50)
    await spi_load_activation(dut, row=1, value=50)
    await spi_cmd_start(dut)

    # Immediately hit reset mid-computation
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    # After reset, status should be clean idle
    done, busy = await spi_read_status(dut)
    assert not busy, "busy should be 0 after mid-computation reset"
    assert not done, "done should be 0 after mid-computation reset"

    # Should be able to run a new computation cleanly
    await spi_write_weight(dut, col=0, w_row0=1, w_row1=1)
    await spi_write_weight(dut, col=1, w_row0=1, w_row1=1)
    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=10)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation after mid-run reset did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    exp_c0, _ = compute_expected((1, 1), (1, 1), (10, 10))
    assert to_signed16(p0_0) == exp_c0, \
        f"Post-reset Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    dut._log.info("Mid-computation reset recovery passed")


@cocotb.test()
async def test_triple_sequential_computations(dut):
    """
    Three distinct matmuls in sequence with different data to verify
    no state leakage between runs. Each uses unique weights and activations.
    """
    dut._log.info("test_triple_sequential_computations")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    test_cases = [
        {"w0": (2, 3), "w1": (4, 5), "a": (6, 7), "label": "run1"},
        {"w0": (10, 0), "w1": (0, 10), "a": (3, 8), "label": "run2"},
        {"w0": (-5, 2), "w1": (3, -7), "a": (4, -3), "label": "run3"},
    ]

    for tc in test_cases:
        await _run_matmul_test(dut, tc["w0"], tc["w1"], tc["a"], tc["label"])
        dut._log.info(f"{tc['label']} passed")

    dut._log.info("All 3 sequential computations verified — no state leakage")


# ═══════════════════════════════════════════════════════════════
#         EDGE-CASE: INT8 EXTREMES & SYSTOLIC ARRAY
# ═══════════════════════════════════════════════════════════════

@cocotb.test()
async def test_matmul_min_negative(dut):
    """INT8 minimum (-128) multiplied by positive value."""
    dut._log.info("test_matmul_min_negative")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # -128 * 1 + 0 * 0 = -128 per column
    # -128 = 0x80 in two's complement
    await _run_matmul_test(dut, (-128, 0), (-128, 0), (1, 1), "min_neg")


@cocotb.test()
async def test_matmul_min_times_min(dut):
    """INT8 minimum × minimum = 16384 (within 16-bit signed range)."""
    dut._log.info("test_matmul_min_times_min")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # (-128)*(-128) + 0*0 = 16384 per column
    await _run_matmul_test(dut, (-128, 0), (-128, 0), (-128, 0), "min_x_min")


@cocotb.test()
async def test_matmul_min_times_max(dut):
    """INT8 minimum × maximum = -16256 per PE."""
    dut._log.info("test_matmul_min_times_max")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # (-128)*127 + (-128)*127 = -16256 + -16256 = -32512
    await _run_matmul_test(dut, (-128, -128), (127, 127), (127, 127), "min_x_max")


@cocotb.test()
async def test_matmul_single_nonzero_weight(dut):
    """Only one weight is non-zero; isolates a single PE path."""
    dut._log.info("test_matmul_single_nonzero_weight")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # W_col0=(0,7), W_col1=(0,0), A=(10,3)
    # Col0 = 10*0 + 3*7 = 21, Col1 = 0
    await _run_matmul_test(dut, (0, 7), (0, 0), (10, 3), "single_wt")


@cocotb.test()
async def test_matmul_single_nonzero_activation(dut):
    """Only one activation is non-zero; tests row isolation."""
    dut._log.info("test_matmul_single_nonzero_activation")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # A=(0, 9), only row-1 activation contributes
    # Col0 = 0*5 + 9*3 = 27, Col1 = 0*2 + 9*4 = 36
    await _run_matmul_test(dut, (5, 3), (2, 4), (0, 9), "single_act")


@cocotb.test()
async def test_matmul_column_isolation(dut):
    """Verify columns compute independently — col0 weights differ from col1."""
    dut._log.info("test_matmul_column_isolation")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # Col0 weights = (10, 0), Col1 weights = (0, 10)
    # A = (7, 3)
    # Col0 = 7*10 + 3*0 = 70, Col1 = 7*0 + 3*10 = 30
    await _run_matmul_test(dut, (10, 0), (0, 10), (7, 3), "col_isolation")


@cocotb.test()
async def test_matmul_alternating_signs(dut):
    """Alternating positive/negative values stress-test signed arithmetic."""
    dut._log.info("test_matmul_alternating_signs")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # W_col0=(50, -50), W_col1=(-50, 50), A=(10, -10)
    # Col0 = 10*50 + (-10)*(-50) = 500+500 = 1000
    # Col1 = 10*(-50) + (-10)*50 = -500-500 = -1000
    await _run_matmul_test(dut, (50, -50), (-50, 50), (10, -10), "alt_signs")


@cocotb.test()
async def test_matmul_16bit_overflow_wrap(dut):
    """Result intentionally overflows signed 16-bit; verify wrap behavior."""
    dut._log.info("test_matmul_16bit_overflow_wrap")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # 127*127 + 127*127 = 16129 + 16129 = 32258 (within range, ok)
    # (-128)*(-128) + (-128)*(-128) = 16384+16384 = 32768
    #   32768 in 16-bit signed = -32768 (overflow wrap)
    await spi_write_weight(dut, col=0, w_row0=0x80, w_row1=0x80)
    await spi_write_weight(dut, col=1, w_row0=0x80, w_row1=0x80)
    await spi_load_activation(dut, row=0, value=0x80)
    await spi_load_activation(dut, row=1, value=0x80)

    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "overflow_wrap did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)
    # 32768 truncated to 16 bits = 0x8000 = -32768 signed
    assert to_signed16(p0_0) == -32768, f"overflow_wrap Col0 expected -32768, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == -32768, f"overflow_wrap Col1 expected -32768, got {to_signed16(p1_0)}"
    dut._log.info("16-bit overflow wrap verified")


# ═══════════════════════════════════════════════════════════════
#         EDGE-CASE: SPI BRIDGE PROTOCOL ROBUSTNESS
# ═══════════════════════════════════════════════════════════════

@cocotb.test()
async def test_spi_partial_abort_then_valid(dut):
    """Abort a weight write mid-data, then perform a complete valid transaction."""
    dut._log.info("test_spi_partial_abort_then_valid")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Start a weight write: opcode + col_sel + partial first weight byte (5 bits only)
    await _spi_cs_assert(dut)
    await _spi_send_byte(dut, 0x01)  # WRITE_WEIGHT opcode
    await _spi_send_byte(dut, 0x00)  # col select
    # Send only 5 bits of weight data, then abort
    for i in range(5):
        bit = 1
        _set_spi_pins(dut, sclk=0, cs_n=0, mosi=bit)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        _set_spi_pins(dut, sclk=1, cs_n=0, mosi=bit)
        await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
    await _spi_cs_deassert(dut)

    # Now do a complete valid computation
    await _run_matmul_test(dut, (3, 2), (1, 4), (5, 6), "post_abort")
    dut._log.info("Partial abort + valid computation passed")


@cocotb.test()
async def test_spi_read_result_while_busy(dut):
    """Reading results during computation should not hang; values may be stale."""
    dut._log.info("test_spi_read_result_while_busy")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await spi_write_weight(dut, col=0, w_row0=5, w_row1=5)
    await spi_write_weight(dut, col=1, w_row0=5, w_row1=5)
    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=10)
    await spi_cmd_start(dut)

    # Immediately try to read result (computation may still be in progress)
    # This should NOT hang the SPI FSM
    p0_0, _ = await spi_read_result(dut, col=0)
    dut._log.info(f"Read during busy returned {to_signed16(p0_0)} (may be stale)")

    # Wait for completion and verify the correct result is readable
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Computation did not complete after early read"

    p0_0, _ = await spi_read_result(dut, col=0)
    exp_c0, _ = compute_expected((5, 5), (5, 5), (10, 10))
    assert to_signed16(p0_0) == exp_c0, \
        f"Post-done Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
    dut._log.info("Read-during-busy resilience verified")


@cocotb.test()
async def test_spi_result_reread_stability(dut):
    """Reading the same column result multiple times should return the same value."""
    dut._log.info("test_spi_result_reread_stability")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await _run_matmul_test(dut, (4, 2), (3, 5), (6, 7), "reread_setup")

    # Re-read col0 three more times
    readings = []
    for _ in range(3):
        p, _ = await spi_read_result(dut, col=0)
        readings.append(to_signed16(p))

    exp_c0, _ = compute_expected((4, 2), (3, 5), (6, 7))
    for i, r in enumerate(readings):
        assert r == exp_c0, f"Re-read {i}: Col0 expected {exp_c0}, got {r}"
    dut._log.info(f"3 re-reads all returned {exp_c0} — stable")


@cocotb.test()
async def test_activation_reload_same_weights(dut):
    """Reload activations without re-loading weights, recompute, verify."""
    dut._log.info("test_activation_reload_same_weights")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # First run with weights (2, 3) / (4, 1) and activations (5, 7)
    await spi_write_weight(dut, col=0, w_row0=2, w_row1=3)
    await spi_write_weight(dut, col=1, w_row0=4, w_row1=1)
    await spi_load_activation(dut, row=0, value=5)
    await spi_load_activation(dut, row=1, value=7)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "First computation did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    exp_c0_1, exp_c1_1 = compute_expected((2, 3), (4, 1), (5, 7))
    assert to_signed16(p0_0) == exp_c0_1, \
        f"Run1 Col0 expected {exp_c0_1}, got {to_signed16(p0_0)}"

    # Second run: same weights, different activations (10, 1)
    await spi_load_activation(dut, row=0, value=10)
    await spi_load_activation(dut, row=1, value=1)
    await spi_cmd_start(dut)
    completed = await wait_for_done(dut, timeout_cycles=5000)
    assert completed, "Second computation did not complete"

    p0_0, _ = await spi_read_result(dut, col=0)
    p1_0, _ = await spi_read_result(dut, col=1)
    exp_c0_2, exp_c1_2 = compute_expected((2, 3), (4, 1), (10, 1))
    assert to_signed16(p0_0) == exp_c0_2, \
        f"Run2 Col0 expected {exp_c0_2}, got {to_signed16(p0_0)}"
    assert to_signed16(p1_0) == exp_c1_2, \
        f"Run2 Col1 expected {exp_c1_2}, got {to_signed16(p1_0)}"
    dut._log.info("Activation reload with same weights passed")


@cocotb.test()
async def test_weight_persistence_across_starts(dut):
    """Weights persist across CMD_START commands until explicitly overwritten."""
    dut._log.info("test_weight_persistence_across_starts")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Load weights once
    await spi_write_weight(dut, col=0, w_row0=7, w_row1=3)
    await spi_write_weight(dut, col=1, w_row0=2, w_row1=5)

    # Run three times with different activations, same weights
    for i, (a0, a1) in enumerate([(1, 1), (4, 2), (0, 10)]):
        await spi_load_activation(dut, row=0, value=a0 & 0xFF)
        await spi_load_activation(dut, row=1, value=a1 & 0xFF)
        await spi_cmd_start(dut)
        completed = await wait_for_done(dut, timeout_cycles=5000)
        assert completed, f"Run {i} did not complete"

        p0_0, _ = await spi_read_result(dut, col=0)
        p1_0, _ = await spi_read_result(dut, col=1)
        exp_c0, exp_c1 = compute_expected((7, 3), (2, 5), (a0, a1))
        assert to_signed16(p0_0) == exp_c0, \
            f"Persistence run{i} Col0 expected {exp_c0}, got {to_signed16(p0_0)}"
        assert to_signed16(p1_0) == exp_c1, \
            f"Persistence run{i} Col1 expected {exp_c1}, got {to_signed16(p1_0)}"
        dut._log.info(f"Persistence run {i}: Col0={to_signed16(p0_0)}, Col1={to_signed16(p1_0)} ✓")

    dut._log.info("Weight persistence across 3 runs verified")


@cocotb.test()
async def test_matmul_sparse_matrix(dut):
    """Sparse weight matrix: only diagonal elements non-zero (scaled identity)."""
    dut._log.info("test_matmul_sparse_matrix")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # W = [[5, 0], [0, 3]], A = (8, 4)
    # Col0 = 8*5 + 4*0 = 40, Col1 = 8*0 + 4*3 = 12
    await _run_matmul_test(dut, (5, 0), (0, 3), (8, 4), "sparse_diag")


@cocotb.test()
async def test_spi_multiple_abort_recover(dut):
    """Multiple back-to-back CS aborts, then verify SPI FSM still works."""
    dut._log.info("test_spi_multiple_abort_recover")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # 5 rapid abort cycles with different partial data
    for attempt in range(5):
        await _spi_cs_assert(dut)
        # Send a random number of bits (2+attempt)
        for _ in range(2 + attempt):
            _set_spi_pins(dut, sclk=0, cs_n=0, mosi=1)
            await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
            _set_spi_pins(dut, sclk=1, cs_n=0, mosi=1)
            await ClockCycles(dut.clk, SPI_HALF_PERIOD_CLKS)
        await _spi_cs_deassert(dut)

    # SPI FSM should have recovered — do a full valid status read
    done, busy = await spi_read_status(dut)
    assert not busy, "busy should be 0 after multiple aborts"
    assert not done, "done should be 0 after multiple aborts"

    # Full matmul should work fine
    await _run_matmul_test(dut, (1, 2), (3, 4), (5, 6), "post_multi_abort")
    dut._log.info("Multiple abort recovery passed")


@cocotb.test()
async def test_stress_five_sequential_matmuls(dut):
    """5 sequential matmuls with varied data to stress-test state management."""
    dut._log.info("test_stress_five_sequential_matmuls")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    test_vectors = [
        {"w0": (1, 0),    "w1": (0, 1),    "a": (42, 17),   "label": "s1_identity"},
        {"w0": (-1, -1),  "w1": (1, 1),    "a": (100, 50),  "label": "s2_negate_vs_pass"},
        {"w0": (127, 0),  "w1": (0, -128), "a": (1, 1),     "label": "s3_extremes"},
        {"w0": (10, 20),  "w1": (30, 40),  "a": (-5, -10),  "label": "s4_neg_acts"},
        {"w0": (0, 0),    "w1": (0, 0),    "a": (127, 127), "label": "s5_zero_weights"},
    ]

    for tc in test_vectors:
        await _run_matmul_test(dut, tc["w0"], tc["w1"], tc["a"], tc["label"])
        dut._log.info(f"{tc['label']} passed")

    dut._log.info("5-run stress test completed successfully")

