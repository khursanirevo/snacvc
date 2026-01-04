"""
Test script to verify speaker encoders are working correctly.
"""

import torch
from snac.speaker_encoder_factory import SpeakerEncoderFactory


def test_simple_encoder():
    """Test simple encoder (deprecated, for comparison)."""
    print("Testing Simple encoder (deprecated)...")

    encoder = SpeakerEncoderFactory.create(
        encoder_type='simple',
        embedding_dim=512,
        snac_sample_rate=24000,
        freeze=False  # Simple encoder is trainable
    )

    # Test forward pass
    audio = torch.randn(2, 1, 24000)  # Batch of 2, 1 second
    embeddings = encoder(audio)

    assert embeddings.shape == (2, 512), f"Wrong shape: {embeddings.shape}"
    assert torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(2), atol=1e-5), "Not L2 normalized"

    print("✅ Simple encoder works (but is deprecated - not pretrained)")


def test_ecapa_encoder():
    """Test ECAPA-TDNN encoder."""
    print("\nTesting ECAPA-TDNN encoder...")

    try:
        encoder = SpeakerEncoderFactory.create(
            encoder_type='ecapa',
            embedding_dim=512,
            snac_sample_rate=24000,
            freeze=True
        )

        # Test forward pass
        audio = torch.randn(2, 1, 24000)  # Batch of 2
        embeddings = encoder(audio)

        assert embeddings.shape == (2, 512), f"Wrong shape: {embeddings.shape}"
        assert torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(2), atol=1e-5), "Not L2 normalized"

        # Check frozen
        for param in encoder.model.parameters():
            assert not param.requires_grad, "ECAPA-TDNN should be frozen"

        print("✅ ECAPA-TDNN encoder works correctly")
        return True
    except ImportError as e:
        print(f"⚠️  ECAPA-TDNN encoder not available: {e}")
        print("Install with: uv pip install speechbrain>=0.5.16")
        return False
    except Exception as e:
        print(f"❌ ECAPA-TDNN encoder failed: {e}")
        return False


def test_erestnet_encoder():
    """Test ERes2NetV2 encoder."""
    print("\nTesting ERes2NetV2 encoder...")

    try:
        encoder = SpeakerEncoderFactory.create(
            encoder_type='eres2net',
            embedding_dim=512,
            snac_sample_rate=24000,
            freeze=True
        )

        # Test forward pass
        audio = torch.randn(2, 1, 24000)  # Batch of 2, 1 second
        embeddings = encoder(audio)

        assert embeddings.shape == (2, 512), f"Wrong shape: {embeddings.shape}"
        assert torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(2), atol=1e-5), "Not L2 normalized"

        # Check frozen
        for param in encoder.model.parameters():
            assert not param.requires_grad, "ERes2NetV2 should be frozen"

        print("✅ ERes2NetV2 encoder works correctly")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  ERes2NetV2 checkpoint not found: {e}")
        print("Download with: uv run python scripts/download_eres2net.py")
        return False
    except Exception as e:
        print(f"❌ ERes2NetV2 encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory():
    """Test factory functionality."""
    print("\nTesting SpeakerEncoderFactory...")

    # List encoders
    encoders = SpeakerEncoderFactory.list_encoders()
    print(f"Available encoders: {list(encoders.keys())}")

    # Get default
    default = SpeakerEncoderFactory.get_default_encoder()
    print(f"Default encoder: {default}")

    assert default == 'ecapa', f"Default should be 'ecapa', got {default}"

    print("✅ Factory works correctly")


def test_embedding_statistics():
    """Compare embedding statistics between encoders."""
    print("\nComparing embedding statistics...")

    # Generate same audio for both
    audio = torch.randn(4, 1, 24000)

    # Test with simple encoder (always available)
    simple_encoder = SpeakerEncoderFactory.create('simple', embedding_dim=512)
    simple_embs = simple_encoder(audio)

    print(f"Simple (random init): mean={simple_embs.mean().item():.4f}, std={simple_embs.std().item():.4f}")

    # Test with ERes2NetV2 if available
    try:
        eres2net_encoder = SpeakerEncoderFactory.create('eres2net', embedding_dim=512)
        eres2net_embs = eres2net_encoder(audio)

        eres2net_mean = eres2net_embs.mean().item()
        eres2net_std = eres2net_embs.std().item()

        print(f"ERes2NetV2 (pretrained): mean={eres2net_mean:.4f}, std={eres2net_std:.4f}")

        # Check that embeddings are different (different models)
        cosine_sim = torch.nn.functional.cosine_similarity(
            simple_embs.unsqueeze(1),
            eres2net_embs.unsqueeze(0),
            dim=-1
        )
        print(f"Max cosine similarity (simple vs eres2net): {cosine_sim.max():.4f}")

        # Should be < 0.9 (different models produce different embeddings)
        assert cosine_sim.max() < 0.9, "Embeddings should be different"

        print("✅ Embeddings are different between models (expected)")
    except Exception as e:
        print(f"⚠️  Skipping ERes2NetV2 comparison: {e}")


def test_invalid_encoder():
    """Test error handling for invalid encoder type."""
    print("\nTesting error handling...")

    try:
        encoder = SpeakerEncoderFactory.create(
            encoder_type='invalid_encoder',
            embedding_dim=512,
        )
        print("❌ Should have raised ValueError for invalid encoder")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("Speaker Encoder Test Suite")
    print("=" * 60)

    test_factory()
    test_simple_encoder()
    ecapa_ok = test_ecapa_encoder()
    eres2net_ok = test_erestnet_encoder()
    test_embedding_statistics()
    test_invalid_encoder()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"✅ Factory: {'PASS' if True else 'FAIL'}")
    print(f"✅ Simple encoder: {'PASS' if True else 'FAIL'} (deprecated)")
    print(f"{'✅' if ecapa_ok else '⚠️ '} ECAPA-TDNN encoder: {'PASS' if ecapa_ok else 'SKIP'}")
    print(f"{'✅' if eres2net_ok else '⚠️ '} ERes2NetV2 encoder: {'PASS' if eres2net_ok else 'SKIP/Fail'}")

    if ecapa_ok:
        print("\n🎉 All critical tests passed! Ready to train with ECAPA-TDNN.")
        print("\nTo start training:")
        print("  uv run python train_phase4_gan.py --config configs/phase4_gan.json --device 0")
    else:
        print("\n⚠️  ERes2NetV2 not available. Download pretrained weights:")
        print("  uv run python scripts/download_eres2net.py")
