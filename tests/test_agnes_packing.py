"""Released Agnes multimodal token layout contracts."""

import torch

from mirai.core.models.agnes_video.packing import pack_inputs


def test_release_input_unfold_and_output_patch_layout():
    video = torch.arange(32.0).reshape(1, 2, 1, 4, 4)
    audio = torch.tensor([[[100.0, 101.0], [102.0, 103.0]]])
    text = torch.ones(1, 3, 12)
    packed = pack_inputs(video, audio, text)
    assert packed.tokens[0, :8].tolist() == [0, 1, 4, 5, 16, 17, 20, 21]
    assert packed.modalities.tolist() == [0]*4 + [1]*2 + [2]*3
    assert packed.cu_seqlens.tolist() == [0, 9]
    prediction = torch.arange(9*8.0).reshape(9, 8)
    restored, restored_audio = packed.unpack(prediction)
    assert restored.shape == video.shape
    assert restored[0, :, 0, :2, :2].tolist() == [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]
    torch.testing.assert_close(restored_audio[0], prediction[4:6, :2])


def test_variable_lengths_and_batch_boundaries():
    packed = pack_inputs(torch.ones(2, 2, 1, 2, 2), torch.ones(2, 5, 2),
                         torch.ones(2, 4, 12), audio_lengths=torch.tensor([3, 5]),
                         text_lengths=torch.tensor([4, 2]))
    assert packed.cu_seqlens.tolist() == [0, 8, 16]
    assert packed.coordinates[4:8, 0].tolist() == [-4, -3, -2, -1]
    assert packed.coordinates[4:8, 3].tolist() == [4]*4
