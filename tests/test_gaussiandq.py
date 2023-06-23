import math
from ccns.reduction.gaussiandq import *


def test__vrb():
    l1 = 1500
    l2 = 480
    s1 = 2.54
    s2 = 1.0
    expected_result = 0.6007609599999999
    result = _vrb(l1, l2, s1, s2)
    assert math.isclose(result, expected_result, rel_tol=1e-8)


def test__vrd():
    sig_d = 0.2
    dr = 0.5
    expected_result = 0.060833333333333336
    result = _vrd(sig_d, dr)
    assert math.isclose(result, expected_result, rel_tol=1e-8)


def test_vrg():
    wl = 3.1
    wl_spread = 0.3
    l1 = 1500
    l2 = 480
    expected_result = 0.00014739183587830952
    result = _vrg(wl, wl_spread, l1, l2)
    assert math.isclose(result, expected_result, rel_tol=1e-8)


def test__vr():
    vrb = 0.1
    vrd = 0.2
    vrg = 0.3
    expected_result = 0.6000000000000001
    result = _vr(vrb, vrd, vrg)
    assert math.isclose(result, expected_result, rel_tol=1e-8)


def test__q_variance():
    q0 = 0.005
    vr = 0.2
    r0 = 3.0
    wl_spread = 0.17
    expected_result = 1.2780555555555557e-06
    result = _q_variance(q0, vr, r0, wl_spread)
    assert math.isclose(result, expected_result, rel_tol=1e-8)


def test__fs():
    vrd = 0.1
    r0 = 5.1
    bs = 5.0
    expected_fs = 0.6240851829770749
    expected_delta_r = 0.2236067977499782
    result_fs, result_delta_r = _fs(vrd, r0, bs)
    assert math.isclose(result_fs, expected_fs, rel_tol=1e-8)
    assert math.isclose(result_delta_r, expected_delta_r, rel_tol=1e-8)


def test__fr():
    vrd = 0.1
    r0 = 5.1
    bs = 5.0
    sig_d = 0.2
    expected_fr = 1.023845766643648
    expected_fs = 0.6240851829770749
    expected_delta_r = 0.2236067977499782
    result_fr, result_fs, result_delta_r = _fr(vrd, r0, bs, sig_d)
    assert math.isclose(result_fr, expected_fr, rel_tol=1e-8)
    assert math.isclose(result_fs, expected_fs, rel_tol=1e-8)
    assert math.isclose(result_delta_r, expected_delta_r, rel_tol=1e-8)


def test__fv():
    vrd = 0.1
    r0 = 5.1
    bs = 5.0
    sig_d = 0.2
    expected_fv = -0.03850469986262093
    result_fv = _fv(vrd, r0, bs, sig_d)
    assert math.isclose(result_fv, expected_fv, rel_tol=1e-8)


def test__beam_stop_correction():
    v_rd = 0.1
    r_0 = 5.1
    b_s = 5.0
    sig_d = 0.2
    expected_v_rds = -0.003850469986262093
    result_v_rds = _beam_stop_correction(v_rd, r_0, b_s, sig_d)
    assert math.isclose(result_v_rds, expected_v_rds, rel_tol=1e-8)


def test__second_order_size_effects():
    f_r = 0.1
    r0 = 5.0
    vrs = 0.3
    expected_r_mean = 0.8
    expected_vr = 0.12
    result_r_mean, result_vr = _second_order_size_effects(f_r, r0, vrs)
    assert math.isclose(result_r_mean, expected_r_mean, rel_tol=1e-8)
    assert math.isclose(result_vr, expected_vr, rel_tol=1e-8)


def test__get_q():
    r = 5.0
    sdd = 480
    wl = 3.1
    expected_q = 0.02111199483074497
    result_q = _get_q(r, sdd, wl)
    assert math.isclose(result_q, expected_q, rel_tol=1e-8)
