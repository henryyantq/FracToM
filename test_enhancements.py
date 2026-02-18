"""Smoke tests for FractalGen-inspired enhancements in nn.py"""
import torch
import nn

def test_forward_with_all_enhancements():
    print('Test 1: Forward pass with capacity_schedule=decreasing, guiding_belief, auxiliary_heads')
    model = nn.FracToMNet(
        input_dim=64, hidden_dim=96, mentalizing_depth=3,
        num_bdi_factors=3, blocks_per_column=1, num_heads=4,
        dropout=0.1, drop_path=0.1, num_classes=4,
        causal_model=True, causal_noise_dim=16,
        capacity_schedule='decreasing', guiding_belief=True,
        gist_dim=32, auxiliary_heads=True,
    )
    print(f'  Column dims: {model.column_dims}')
    x = torch.randn(8, 64)
    logits, report = model(x, return_interpretability=True)
    print(f'  Output shape: {logits.shape}')
    print(f'  Aux logits keys: {list(report.auxiliary_logits.keys()) if report.auxiliary_logits else None}')
    print(f'  Guiding gists keys: {list(report.guiding_gists.keys()) if report.guiding_gists else None}')
    print(f'  Column dims in report: {report.column_dims}')
    assert logits.shape == (8, 4)
    assert report.auxiliary_logits is not None
    assert len(report.auxiliary_logits) == 4  # 4 columns
    assert report.guiding_gists is not None
    assert report.column_dims is not None
    print('  PASS\n')

def test_backward_with_aux():
    print('Test 2: Backward pass with FracToMLoss including aux deep supervision')
    model = nn.FracToMNet(
        input_dim=64, hidden_dim=96, mentalizing_depth=3,
        num_bdi_factors=3, blocks_per_column=1, num_heads=4,
        num_classes=4, causal_model=True,
        capacity_schedule='decreasing', guiding_belief=True,
        gist_dim=32, auxiliary_heads=True,
    )
    x = torch.randn(8, 64)
    logits, report = model(x, return_interpretability=True)
    targets = torch.randint(0, 4, (8,))
    criterion = nn.FracToMLoss(lambda_auxiliary=0.1)
    loss, bkd = criterion(logits, targets, report)
    print(f'  Total loss: {loss.item():.4f}')
    print(f'  Aux deepsup: {bkd["aux_deepsup"]:.4f}')
    loss.backward()
    print('  Backward: OK')
    print('  PASS\n')

def test_backward_compat():
    print('Test 3: Backward compatibility (uniform, no guiding, no aux)')
    model = nn.FracToMNet(
        input_dim=64, hidden_dim=96, mentalizing_depth=2,
        num_bdi_factors=3, num_classes=3,
        capacity_schedule='uniform', guiding_belief=False,
        auxiliary_heads=False,
    )
    x = torch.randn(4, 64)
    logits, r = model(x, return_interpretability=True)
    print(f'  Column dims: {model.column_dims}')
    print(f'  Aux logits: {r.auxiliary_logits}')
    print(f'  Guiding gists: {r.guiding_gists}')
    assert all(d == 96 for d in model.column_dims)
    assert r.auxiliary_logits is None
    assert r.guiding_gists is None
    t = torch.randint(0, 3, (4,))
    criterion = nn.FracToMLoss()
    l, b = criterion(logits, t, r)
    l.backward()
    print(f'  Backward: OK, loss={l.item():.4f}')
    print('  PASS\n')

def test_capacity_schedule_math():
    print('Test 4: Capacity schedule math')
    m = nn.FracToMNet(
        input_dim=32, hidden_dim=96, mentalizing_depth=3,
        num_bdi_factors=3, num_classes=2,
        capacity_schedule='decreasing',
    )
    for k, d in enumerate(m.column_dims):
        print(f'  Column {k}: dim={d}')
    assert m.column_dims[0] > m.column_dims[-1], 'dim[0] should be > dim[-1]'
    for d in m.column_dims:
        assert d % 3 == 0, f'dim={d} not divisible by 3'
    print('  PASS\n')

def test_param_count():
    print('Test 5: Param counts (decreasing vs uniform)')
    m_dec = nn.FracToMNet(
        input_dim=32, hidden_dim=96, mentalizing_depth=3, num_classes=2,
        capacity_schedule='decreasing', guiding_belief=False, auxiliary_heads=False,
    )
    m_uni = nn.FracToMNet(
        input_dim=32, hidden_dim=96, mentalizing_depth=3, num_classes=2,
        capacity_schedule='uniform', guiding_belief=False, auxiliary_heads=False,
    )
    p_dec = sum(p.numel() for p in m_dec.parameters())
    p_uni = sum(p.numel() for p in m_uni.parameters())
    print(f'  Decreasing: {p_dec:,} params')
    print(f'  Uniform:    {p_uni:,} params')
    print(f'  Ratio: {p_dec/p_uni:.3f}')
    assert p_dec < p_uni, 'Decreasing should have fewer params'
    print('  PASS\n')

def test_sequence_fractom():
    print('Test 6: SequenceFracToM with enhancements')
    sm = nn.SequenceFracToM(
        vocab_size=500, embed_dim=96, max_seq_len=32,
        seq_encoder_layers=1, num_heads=4,
        mentalizing_depth=2, num_classes=3,
        capacity_schedule='decreasing',
        guiding_belief=True, gist_dim=32,
        auxiliary_heads=True,
    )
    tokens = torch.randint(0, 500, (4, 16))
    out, rep = sm(tokens, return_interpretability=True)
    print(f'  Output shape: {out.shape}')
    loss = out.sum()
    loss.backward()
    print('  Backward: OK')
    print('  PASS\n')

def test_aux_accuracy_check():
    """Verify aux heads produce correct-shape logits and gradients flow."""
    print('Test 7: Auxiliary heads shape & gradient check')
    model = nn.FracToMNet(
        input_dim=32, hidden_dim=96, mentalizing_depth=2,
        num_bdi_factors=3, num_classes=5,
        capacity_schedule='decreasing', auxiliary_heads=True,
    )
    model.eval()  # disable drop-path so all columns get gradients
    x = torch.randn(4, 32)
    logits, report = model(x, return_interpretability=True)
    for k, aux_l in report.auxiliary_logits.items():
        assert aux_l.shape == (4, 5), f'Column {k}: expected (4,5), got {aux_l.shape}'
    # Check gradients flow through aux heads
    targets = torch.randint(0, 5, (4,))
    criterion = nn.FracToMLoss(lambda_auxiliary=1.0)
    loss, bkd = criterion(logits, targets, report)
    loss.backward()
    for k in range(3):  # 3 columns
        grad_norm = model.aux_heads[k].weight.grad.norm().item()
        assert grad_norm > 0, f'No gradient in aux head {k}'
        print(f'  Aux head {k} grad norm: {grad_norm:.4f}')
    print('  PASS\n')

def test_guiding_identity_init():
    """Verify GuidingBeliefModule starts as identity (gamma=1, beta=0)."""
    print('Test 8: GuidingBeliefModule identity initialisation')
    gb = nn.GuidingBeliefModule(input_dim=64, gist_dim=32, output_dim=96)
    x = torch.randn(4, 64)
    gamma, beta = gb(x)
    # With zero-init weights, gamma should be ~1 and beta ~0
    print(f'  gamma mean: {gamma.mean():.4f} (expect ~1.0)')
    print(f'  beta mean: {beta.mean():.4f} (expect ~0.0)')
    # Modulate should approximately preserve input
    h = torch.randn(4, 96)
    h_mod = nn.GuidingBeliefModule.modulate(h, gamma, beta)
    diff = (h_mod - h).abs().mean().item()
    print(f'  Modulation diff from identity: {diff:.6f}')
    assert diff < 0.01, f'Initial modulation should be near-identity, got diff={diff}'
    print('  PASS\n')


if __name__ == '__main__':
    test_forward_with_all_enhancements()
    test_backward_with_aux()
    test_backward_compat()
    test_capacity_schedule_math()
    test_param_count()
    test_sequence_fractom()
    test_aux_accuracy_check()
    test_guiding_identity_init()
    print('=' * 60)
    print('All 8 tests PASSED!')
    print('=' * 60)
