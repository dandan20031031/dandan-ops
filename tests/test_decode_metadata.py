"""CPU coverage for packed metadata; no CUDA tensors or engine import needed."""
from types import SimpleNamespace

import numpy as np
import pytest

from dandan_ops.integrations.myvllm_decode import fill_decode_metadata


def request(token=7, position=0, pages=(1,), *, finished=False, output=True):
    return SimpleNamespace(
        output_token_ids=[token] if output else [],
        total_len=position + 1,
        block_ids=list(pages),
        is_finished=finished,
    )


@pytest.fixture
def config():
    return dict(batch_size=4, max_blocks=4, padding=(56, 57, 58, 59),
                block_size=4, num_blocks=64, vocab_size=1000, max_position=128)


@pytest.fixture
def slab(config):
    return np.full(config["batch_size"] * (2 + config["max_blocks"]),
                   703, dtype=np.int64)


def views(array, config):
    b, w = config["batch_size"], config["max_blocks"]
    return array[:b], array[b:2*b], array[2*b:].reshape(b, w)


def assert_unchanged_on_failure(array, requests, config, **changes):
    before = array.copy()
    with pytest.raises(ValueError):
        fill_decode_metadata(array, requests, **{**config, **changes})
    np.testing.assert_array_equal(array, before)


def test_slab_is_contiguous_tokens_positions_then_row_major_tables(slab, config):
    requests = [request(11, 5, (1, 2)), request(13, 8, (3, 4, 5))]
    fill_decode_metadata(slab, requests, **config)
    tokens, positions, tables = views(slab, config)
    np.testing.assert_array_equal(tokens, [11, 13, 0, 0])
    np.testing.assert_array_equal(positions, [5, 8, 0, 0])
    np.testing.assert_array_equal(tables, [
        [1, 2, -1, -1], [3, 4, 5, -1],
        [58, 57, 58, 59], [59, 57, 58, 59],
    ])
    assert slab.dtype == np.int64 and slab.flags.c_contiguous
    assert all(np.shares_memory(slab, part) for part in (tokens, positions, tables))
    assert tables.flags.c_contiguous
    assert tables.strides == (4 * slab.itemsize, slab.itemsize)
    assert positions.ctypes.data - tokens.ctypes.data == 4 * slab.itemsize
    assert tables.ctypes.data - tokens.ctypes.data == 8 * slab.itemsize


def test_reordered_requests_refill_rows_without_identity_cache(slab, config):
    first = request(11, 5, (1, 2))
    second = request(13, 8, (3, 4, 5))
    fill_decode_metadata(slab, [first, second], **config)
    fill_decode_metadata(slab, [second, first], **config)
    tokens, positions, tables = views(slab, config)
    np.testing.assert_array_equal(tokens[:2], [13, 11])
    np.testing.assert_array_equal(positions[:2], [8, 5])
    np.testing.assert_array_equal(tables[:2], [[3, 4, 5, -1], [1, 2, -1, -1]])


def test_shrinking_then_growing_batch_restores_private_padding(slab, config):
    first = request(11, 5, (1, 2))
    second = request(13, 8, (3, 4, 5))
    third = request(17, 4, (6, 7))
    fill_decode_metadata(slab, [first, second, third], **config)
    fill_decode_metadata(slab, [second], **config)
    tokens, positions, tables = views(slab, config)
    np.testing.assert_array_equal(tokens, [13, 0, 0, 0])
    np.testing.assert_array_equal(positions, [8, 0, 0, 0])
    np.testing.assert_array_equal(tables[1:, 0], [57, 58, 59])
    assert not set(tables[1:].ravel()).intersection({1, 2, 3, 4, 5, 6, 7})
    fill_decode_metadata(slab, [third, first, second], **config)
    np.testing.assert_array_equal(tokens, [17, 11, 13, 0])
    np.testing.assert_array_equal(tables[:3], [
        [6, 7, -1, -1], [1, 2, -1, -1], [3, 4, 5, -1],
    ])


def test_page_growth_refreshes_same_request_and_clears_old_tail(slab, config):
    req = request(19, 7, (1, 2))
    fill_decode_metadata(slab, [req], **config)
    req.output_token_ids.append(23)
    req.total_len = 9
    req.block_ids.append(6)
    fill_decode_metadata(slab, [req], **config)
    tokens, positions, tables = views(slab, config)
    assert tokens[0] == 23 and positions[0] == 8
    np.testing.assert_array_equal(tables[0], [1, 2, 6, -1])
    # A shorter replacement reuses row zero; prior physical pages must vanish.
    fill_decode_metadata(slab, [request(29, 0, (9,))], **config)
    np.testing.assert_array_equal(tables[0], [9, -1, -1, -1])


def test_inactive_rows_have_distinct_private_write_pages(slab, config):
    fill_decode_metadata(slab, [request(31, 0, (1,))], **config)
    tokens, positions, tables = views(slab, config)
    targets = [(int(tables[row, positions[row] // config["block_size"]]),
                int(positions[row] % config["block_size"]))
               for row in range(config["batch_size"])]
    assert len(set(targets)) == config["batch_size"]
    assert targets == [(1, 0), (57, 0), (58, 0), (59, 0)]
    assert np.all(tokens[1:] == 0) and np.all(positions[1:] == 0)


def test_shared_readonly_prefix_is_allowed_with_private_write_pages(slab, config):
    fill_decode_metadata(slab, [
        request(37, 8, (1, 2, 3)), request(41, 9, (1, 2, 4)),
    ], **config)
    _, positions, tables = views(slab, config)
    np.testing.assert_array_equal(tables[:2, :2], [[1, 2], [1, 2]])
    assert tables[0, positions[0] // 4] == 3
    assert tables[1, positions[1] // 4] == 4


@pytest.mark.parametrize("requests", [
    [request(1, 4, (1, 2)), request(2, 4, (3, 2))],  # Same slot.
    [request(1, 4, (1, 2)), request(2, 5, (3, 2))],  # Same page, different slot.
    [request(1, 4, (1, 2)), request(2, 8, (2, 3, 4))],  # Writer aliases other's prefix.
    [request(1, 8, (2, 3, 4)), request(2, 4, (1, 2))],  # Reverse order must also fail.
    [request(1, 4, (2, 2))],  # Writer aliases its own earlier logical page.
])
def test_write_pages_are_exclusive_to_their_request(slab, config, requests):
    assert_unchanged_on_failure(slab, requests, config)


@pytest.mark.parametrize("invalid", [
    request(token=-1), request(token=1000), request(token=1.5), request(token=True),
    request(position=-1), request(position=128, pages=(1, 2, 3, 4)),
    request(position=0.5), request(pages=()),
    request(position=4, pages=(1,)), request(pages=(1, 2, 3, 4, 5)),
    request(pages=(-1,)), request(pages=(64,)), request(pages=(1.5,)),
    request(pages=(True,)), request(pages=(56,)),
    request(finished=True), request(output=False),
])
def test_invalid_request_does_not_partially_change_slab(slab, config, invalid):
    # Validate a valid first row and an invalid later row before modifying any row.
    assert_unchanged_on_failure(slab, [request(43, 0, (12,)), invalid], config)


@pytest.mark.parametrize("padding", [
    (56, 57, 58), (56, 56, 58, 59), (-1, 57, 58, 59),
    (64, 57, 58, 59), (56.5, 57, 58, 59), (True, 57, 58, 59),
])
def test_invalid_scratch_configuration_keeps_slab_unchanged(slab, config, padding):
    assert_unchanged_on_failure(slab, [request()], config, padding=padding)


def test_overfull_batch_keeps_slab_unchanged(slab, config):
    assert_unchanged_on_failure(slab, [request(pages=(p,)) for p in range(1, 6)], config)


def test_failure_preserves_previously_valid_metadata_not_only_sentinel(slab, config):
    fill_decode_metadata(slab, [request(47, 4, (10, 11))], **config)
    assert_unchanged_on_failure(slab, [request(53, 0, (8,)), request(token=1000)], config)


def test_numpy_integer_metadata_is_accepted(slab, config):
    req = request(np.int64(59), np.int64(5), (np.int64(1), np.int64(2)))
    fill_decode_metadata(slab, [req], **config)
    tokens, positions, tables = views(slab, config)
    assert tokens[0] == 59 and positions[0] == 5
    np.testing.assert_array_equal(tables[0], [1, 2, -1, -1])
