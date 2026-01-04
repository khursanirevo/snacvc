#!/usr/bin/env python3
"""
Integration helper to add diagnostics to existing training script.

This script shows how to modify train_phase4_gan.py to include diagnostics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Add this to train_phase4_gan.py imports:
"""
# Add near the top of train_phase4_gan.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

from scripts.diagnostics.monitor_training import (
    compute_gradient_norm,
    compute_parameter_norm,
    compute_speaker_similarity,
    check_gan_health,
    generate_diagnostics_report,
)
from scripts.diagnostics.generate_samples import (
    generate_reconstruction,
    generate_voice_conversion,
    generate_evaluation_samples,
)
"""

# Example: Add diagnostics to train_epoch function:
DIAGNOSTICS_INTEGRATION_EXAMPLE = """
def train_epoch(model, mpd, mrd, dataloader, opt_gen, opt_disc, device, config,
                output_dir=None, epoch=0, start_step=0,
                scheduler_gen=None, scheduler_disc=None,
                use_ddp=False, run_diagnostics=False):
    \"\"\"
    Train for one epoch with optional diagnostics.

    New args:
        run_diagnostics: If True, run comprehensive diagnostics every N batches
    \"\"\"

    # ... existing setup code ...

    # Add: Diagnostics tracking
    prev_losses = None
    diagnostics_interval = config.get('diagnostics_interval', 100)

    for batch_idx, batch in enumerate(pbar):
        # ... existing training code ...

        # ========== NEW: Run diagnostics periodically ==========
        if run_diagnostics and batch_idx % diagnostics_interval == 0 and batch_idx > 0:
            print(f"\\n[Running Diagnostics - Step {start_step + batch_idx}]")

            # Prepare fake audio for similarity check
            with torch.no_grad():
                model_base = model.module if use_ddp else model
                fake_audio_for_diag = audio_hat.detach()

            diagnostics_batch = {
                'real_audio': audio,
                'fake_audio': fake_audio_for_diag,
            }

            metrics, status = generate_diagnostics_report(
                model=model,
                mpd=mpd,
                mrd=mrd,
                batch=diagnostics_batch,
                device=device,
                losses={
                    'gen': loss_gen.item(),
                    'disc': loss_disc.item(),
                    'recon': loss_recon.item(),
                    'contrast': loss_contrast.item(),
                    'adv': loss_adv.item(),
                    'fm': loss_fm.item(),
                },
                prev_losses=prev_losses,
                step=start_step + batch_idx,
                output_dir=f"{output_dir}/diagnostics" if output_dir else None
            )

            # Store for next iteration
            prev_losses = {
                'gen': loss_gen.item(),
                'disc': loss_disc.item(),
                'recon': loss_recon.item(),
                'contrast': loss_contrast.item(),
            }

            # Check if critical
            if status == 'critical':
                print("\\n⚠️ CRITICAL: Training health issues detected!")
                print("Consider: reducing learning rate, adjusting loss weights, or checking data")
        # ========== END Diagnostics ==========

        # ... rest of training loop ...
"""

# Example: Add sample generation during validation:
SAMPLE_GENERATION_EXAMPLE = """
def evaluate_with_samples(model, mpd, mrd, dataloader, device, config,
                          output_dir, epoch, use_ddp=False):
    \"\"\"
    Evaluation with sample generation for inspection.
    \"\"\"
    # ... existing evaluation code ...

    # ========== NEW: Generate samples every N epochs ==========
    if epoch % config.get('sample_interval', 5) == 0:
        print("\\n[Generating Evaluation Samples]")

        sample_dir = Path(output_dir) / "samples" / f"epoch_{epoch}"
        generate_evaluation_samples(
            model=model,
            dataset=dataloader.dataset,  # Assumes dataset has .dataset attribute
            device=device,
            output_dir=sample_dir,
            num_samples=10,
        )

        print(f"Samples saved to: {sample_dir}")
    # ========== END Sample Generation ==========
"""

# Example: Main loop modifications:
MAIN_LOOP_EXAMPLE = """
# In main() function, add these parameters:
parser.add_argument('--run-diagnostics', action='store_true',
                    help='Enable comprehensive diagnostics during training')
parser.add_argument('--diagnostics-interval', type=int, default=100,
                    help='Run diagnostics every N batches')
parser.add_argument('--sample-interval', type=int, default=5,
                    help='Generate samples every N epochs')

# When calling train_epoch:
train_metrics = train_epoch(
    model, mpd, mrd, train_loader, opt_gen, opt_disc, device, config,
    output_dir=output_dir, epoch=epoch, start_step=start_step,
    scheduler_gen=scheduler_gen, scheduler_disc=scheduler_disc,
    use_ddp=use_ddp,
    run_diagnostics=args.run_diagnostics,  # NEW
)

# When calling evaluate:
val_metrics = evaluate_with_samples(
    model, mpd, mrd, val_loader, device, config,
    output_dir=output_dir, epoch=epoch,
    use_ddp=use_ddp,
)
"""

if __name__ == "__main__":
    print("="*70)
    print("Diagnostics Integration Guide")
    print("="*70)
    print("\nTo integrate diagnostics into train_phase4_gan.py:")
    print("\n1. Add the imports (see DIAGNOSTICS_INTEGRATION_EXAMPLE)")
    print("2. Modify train_epoch to include diagnostics")
    print("3. Modify evaluate to include sample generation")
    print("4. Add CLI arguments to main()")
    print("\nSee the examples above for code snippets.")
    print("\nQuick integration:")
    print("  --run-diagnostics: Enable diagnostics during training")
    print("  --diagnostics-interval N: Run diagnostics every N batches")
    print("  --sample-interval N: Generate samples every N epochs")
    print("="*70)
