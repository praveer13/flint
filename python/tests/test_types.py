"""Tests for flint_shm types — verify numpy dtypes match Zig extern struct layouts.

These tests ensure that the Python dtype definitions are byte-compatible with
the Zig extern structs in src/shm/types.zig. If any test here fails, the
shared memory IPC will silently produce garbage data.
"""

import mmap
import struct

import numpy as np
import pytest

from flint_shm.types import (
    MAX_BATCH,
    MAX_BLOCKS_PER_SEQ,
    MAX_SWAP,
    COMPLETION_DTYPE,
    SWAP_CMD_DTYPE,
    SCHEDULE_DTYPE,
)
from flint_shm.ring_buffer import RingReader, RingWriter, HEADER_SIZE
from flint_shm.heartbeat import HeartbeatRegion


# ---------------------------------------------------------------------------
# CompletionT tests
# ---------------------------------------------------------------------------

class TestCompletionDtype:
    def test_size_is_16_bytes(self):
        assert COMPLETION_DTYPE.itemsize == 16

    def test_field_offsets(self):
        """Offsets must match Zig comptime assertions in types.zig."""
        fields = COMPLETION_DTYPE.fields
        assert fields['seq_id'][1] == 0
        assert fields['token_id'][1] == 8
        assert fields['logprob'][1] == 12
        assert fields['is_eos'][1] == 14
        assert fields['_pad'][1] == 15

    def test_roundtrip_values(self):
        """Create a record, set fields, read them back."""
        c = np.zeros(1, dtype=COMPLETION_DTYPE)[0]
        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        arr[0]['seq_id'] = 0xDEAD_BEEF_CAFE_BABE
        arr[0]['token_id'] = 31337
        arr[0]['logprob'] = np.float16(-2.5)
        arr[0]['is_eos'] = 1
        c = arr[0]
        assert int(c['seq_id']) == 0xDEAD_BEEF_CAFE_BABE
        assert int(c['token_id']) == 31337
        assert float(c['logprob']) == pytest.approx(-2.5)
        assert int(c['is_eos']) == 1

    def test_raw_bytes_interpretation(self):
        """Verify byte-level layout matches Zig's test in types.zig."""
        buf = bytearray(16)
        # seq_id = 1 at offset 0 (little-endian u64)
        buf[0] = 1
        # token_id = 256 at offset 8 (little-endian u32)
        buf[8] = 0
        buf[9] = 1
        # is_eos = 1 at offset 14
        buf[14] = 1

        c = np.frombuffer(buf, dtype=COMPLETION_DTYPE, count=1)[0]
        assert int(c['seq_id']) == 1
        assert int(c['token_id']) == 256
        assert int(c['is_eos']) == 1


# ---------------------------------------------------------------------------
# SwapCmd tests
# ---------------------------------------------------------------------------

class TestSwapCmdDtype:
    def test_size_is_8_bytes(self):
        assert SWAP_CMD_DTYPE.itemsize == 8

    def test_field_offsets(self):
        fields = SWAP_CMD_DTYPE.fields
        assert fields['gpu_block_id'][1] == 0
        assert fields['cpu_block_id'][1] == 4

    def test_roundtrip(self):
        arr = np.zeros(1, dtype=SWAP_CMD_DTYPE)
        arr[0]['gpu_block_id'] = 100
        arr[0]['cpu_block_id'] = 200
        assert int(arr[0]['gpu_block_id']) == 100
        assert int(arr[0]['cpu_block_id']) == 200


# ---------------------------------------------------------------------------
# ScheduleT tests
# ---------------------------------------------------------------------------

class TestScheduleDtype:
    def test_size_is_multiple_of_8(self):
        """Zig asserts @sizeOf(ScheduleT) % 8 == 0."""
        assert SCHEDULE_DTYPE.itemsize % 8 == 0

    def test_expected_size(self):
        """Verify total size matches manual calculation from Zig layout."""
        expected = 533792
        assert SCHEDULE_DTYPE.itemsize == expected

    def test_key_field_offsets(self):
        """Offsets must match Zig comptime assertions."""
        fields = SCHEDULE_DTYPE.fields
        assert fields['iteration_id'][1] == 0
        assert fields['num_sequences'][1] == 8
        assert fields['_align_pad'][1] == 12
        assert fields['seq_ids'][1] == 16

    def test_all_field_offsets(self):
        """Verify every field offset against the Zig layout."""
        fields = SCHEDULE_DTYPE.fields
        assert fields['iteration_id'][1] == 0
        assert fields['num_sequences'][1] == 8
        assert fields['_align_pad'][1] == 12
        assert fields['seq_ids'][1] == 16
        assert fields['token_ids'][1] == 16 + MAX_BATCH * 8      # 2064
        assert fields['positions'][1] == 2064 + MAX_BATCH * 4     # 3088
        assert fields['seq_lens'][1] == 3088 + MAX_BATCH * 4      # 4112
        assert fields['is_prefill'][1] == 4112 + MAX_BATCH * 4    # 5136
        assert fields['temperatures'][1] == 5136 + MAX_BATCH * 1  # 5392
        assert fields['top_ps'][1] == 5392 + MAX_BATCH * 2        # 5904
        assert fields['max_tokens'][1] == 5904 + MAX_BATCH * 2    # 6416
        assert fields['block_tables'][1] == 6416 + MAX_BATCH * 4  # 7440
        assert fields['num_blocks'][1] == 7440 + MAX_BATCH * MAX_BLOCKS_PER_SEQ * 4  # 531728
        assert fields['num_swap_out'][1] == 531728 + MAX_BATCH * 4  # 532752
        assert fields['num_swap_in'][1] == 532756
        assert fields['swap_out'][1] == 532760
        assert fields['swap_in'][1] == 532760 + MAX_SWAP * 8      # 533272
        assert fields['is_shutdown'][1] == 533272 + MAX_SWAP * 8   # 533784
        assert fields['_pad'][1] == 533785

    def test_create_and_populate(self):
        """Create a schedule, populate fields, verify readback."""
        arr = np.zeros(1, dtype=SCHEDULE_DTYPE)
        arr[0]['iteration_id'] = 42
        arr[0]['num_sequences'] = 3
        arr[0]['seq_ids'][0] = 100
        arr[0]['seq_ids'][1] = 200
        arr[0]['seq_ids'][2] = 300
        arr[0]['token_ids'][0] = 10
        arr[0]['positions'][0] = 50
        arr[0]['seq_lens'][0] = 51
        arr[0]['is_prefill'][0] = 1
        arr[0]['temperatures'][0] = np.float16(0.7)
        arr[0]['top_ps'][0] = np.float16(0.9)
        arr[0]['max_tokens'][0] = 1024
        arr[0]['num_blocks'][0] = 3
        arr[0]['block_tables'][0][0] = 10
        arr[0]['block_tables'][0][1] = 20
        arr[0]['block_tables'][0][2] = 30
        arr[0]['num_swap_out'] = 1
        arr[0]['swap_out'][0]['gpu_block_id'] = 5
        arr[0]['swap_out'][0]['cpu_block_id'] = 50
        arr[0]['is_shutdown'] = 0

        s = arr[0]
        assert int(s['iteration_id']) == 42
        assert int(s['num_sequences']) == 3
        assert int(s['seq_ids'][1]) == 200
        assert int(s['token_ids'][0]) == 10
        assert int(s['positions'][0]) == 50
        assert int(s['seq_lens'][0]) == 51
        assert int(s['is_prefill'][0]) == 1
        assert float(s['temperatures'][0]) == pytest.approx(0.7, abs=0.01)
        assert float(s['top_ps'][0]) == pytest.approx(0.9, abs=0.01)
        assert int(s['max_tokens'][0]) == 1024
        assert int(s['num_blocks'][0]) == 3
        assert int(s['block_tables'][0][1]) == 20
        assert int(s['num_swap_out']) == 1
        assert int(s['swap_out'][0]['gpu_block_id']) == 5
        assert int(s['swap_out'][0]['cpu_block_id']) == 50

    def test_shutdown_flag(self):
        arr = np.zeros(1, dtype=SCHEDULE_DTYPE)
        arr[0]['is_shutdown'] = 1
        assert int(arr[0]['is_shutdown']) == 1


# ---------------------------------------------------------------------------
# Ring buffer tests (using anonymous mmap)
# ---------------------------------------------------------------------------

def _make_anon_mmap(size: int) -> mmap.mmap:
    """Create an anonymous mmap region (no file backing) for testing."""
    return mmap.mmap(-1, size)


class TestRingBuffer:
    def test_push_pop_completion(self):
        """Push one CompletionT, pop it back, verify fields."""
        capacity = 8
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)

        writer = RingWriter(mm, 0, capacity, COMPLETION_DTYPE)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)

        # Create a completion record
        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        arr[0]['seq_id'] = 42
        arr[0]['token_id'] = 1234
        arr[0]['logprob'] = np.float16(-1.5)
        arr[0]['is_eos'] = 1

        assert writer.try_push(arr[0]) is True
        result = reader.try_pop()
        assert result is not None
        assert int(result['seq_id']) == 42
        assert int(result['token_id']) == 1234
        assert float(result['logprob']) == pytest.approx(-1.5)
        assert int(result['is_eos']) == 1

    def test_empty_pop_returns_none(self):
        capacity = 4
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)
        assert reader.try_pop() is None

    def test_full_ring_rejects_push(self):
        capacity = 4
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        writer = RingWriter(mm, 0, capacity, COMPLETION_DTYPE)

        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        for i in range(capacity):
            arr[0]['seq_id'] = i
            assert writer.try_push(arr[0]) is True

        # Ring is full
        arr[0]['seq_id'] = 999
        assert writer.try_push(arr[0]) is False

    def test_fifo_order(self):
        """Push multiple items, verify they come out in order."""
        capacity = 8
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        writer = RingWriter(mm, 0, capacity, COMPLETION_DTYPE)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)

        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        for i in range(5):
            arr[0]['seq_id'] = i * 10
            arr[0]['token_id'] = i
            writer.try_push(arr[0])

        for i in range(5):
            result = reader.try_pop()
            assert result is not None
            assert int(result['seq_id']) == i * 10
            assert int(result['token_id']) == i

    def test_wraparound(self):
        """Fill and drain twice to exercise modulo wraparound."""
        capacity = 4
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        writer = RingWriter(mm, 0, capacity, COMPLETION_DTYPE)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)

        arr = np.zeros(1, dtype=COMPLETION_DTYPE)

        # First pass
        for i in range(capacity):
            arr[0]['token_id'] = i
            writer.try_push(arr[0])
        for i in range(capacity):
            result = reader.try_pop()
            assert int(result['token_id']) == i

        # Second pass (head and tail are now >= capacity)
        for i in range(capacity):
            arr[0]['token_id'] = 100 + i
            writer.try_push(arr[0])
        for i in range(capacity):
            result = reader.try_pop()
            assert int(result['token_id']) == 100 + i

    def test_len(self):
        capacity = 8
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        writer = RingWriter(mm, 0, capacity, COMPLETION_DTYPE)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)

        assert reader.len() == 0
        assert writer.len() == 0

        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        for i in range(3):
            arr[0]['seq_id'] = i
            writer.try_push(arr[0])

        assert reader.len() == 3

        reader.try_pop()
        assert reader.len() == 2

    def test_wait_pop_timeout(self):
        capacity = 4
        ring_size = HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        reader = RingReader(mm, 0, capacity, COMPLETION_DTYPE)

        with pytest.raises(TimeoutError):
            reader.wait_pop(timeout_ms=50)

    def test_nonzero_offset(self):
        """Ring buffer at a non-zero offset within the mmap region."""
        offset = 256  # Some arbitrary offset
        capacity = 4
        ring_size = offset + HEADER_SIZE + COMPLETION_DTYPE.itemsize * capacity
        mm = _make_anon_mmap(ring_size)
        writer = RingWriter(mm, offset, capacity, COMPLETION_DTYPE)
        reader = RingReader(mm, offset, capacity, COMPLETION_DTYPE)

        arr = np.zeros(1, dtype=COMPLETION_DTYPE)
        arr[0]['seq_id'] = 77
        arr[0]['token_id'] = 88
        assert writer.try_push(arr[0]) is True

        result = reader.try_pop()
        assert result is not None
        assert int(result['seq_id']) == 77
        assert int(result['token_id']) == 88


# ---------------------------------------------------------------------------
# Heartbeat tests
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_increment_and_read(self):
        mm = _make_anon_mmap(64)
        hb = HeartbeatRegion(mm, 0)
        assert hb.read() == 0
        hb.increment()
        assert hb.read() == 1
        hb.increment()
        hb.increment()
        assert hb.read() == 3

    def test_write(self):
        mm = _make_anon_mmap(64)
        hb = HeartbeatRegion(mm, 0)
        hb.write(42)
        assert hb.read() == 42

    def test_nonzero_offset(self):
        mm = _make_anon_mmap(128)
        hb = HeartbeatRegion(mm, 64)
        hb.increment()
        assert hb.read() == 1
        # Verify offset 0 is untouched
        assert struct.unpack_from('<Q', mm, 0)[0] == 0
