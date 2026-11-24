//! The two compute passes that expand indirect rectangle-list draws
//! (`victory-gpu`'s `rectangles` says why and when they run), built with our
//! own writer so that no system shader compiler is needed: Wine's
//! `D3DCompile` has no `RWByteAddressBuffer::Load`.
//!
//! Both follow the HLSL they replace line for line (kept in `victory-gpu`'s
//! history): [`rect_rewrite`] rewrites each record into a draw of six
//! vertices per rectangle and reserves room for them, and [`rect_expand`]
//! copies each rectangle's three vertices into six and derives the fourth
//! corner's elements as `v0 + v2 - v1`.
//!
//! Every root view is at the register of its root parameter, in
//! [`super::ROOT_CBV_SPACE`]; the user data (eight dwords) is parameter 0.

use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::shader::Shader;

/// [`rect_rewrite`]'s raw buffers (root UAVs): the title's clamped
/// records, the count buffer, the vertex allocator, the rewritten records.
pub const REWRITE_SRC_ARGS: u8 = 1;
pub const REWRITE_COUNT: u8 = 2;
pub const REWRITE_ALLOC: u8 = 3;
pub const REWRITE_DST_ARGS: u8 = 4;
/// [`rect_expand`]'s element table (a root SRV) and raw buffers (root
/// UAVs): records as the title wrote them, as rewritten, the source and the
/// expanded vertices.
pub const EXPAND_ELEMS: u8 = 1;
pub const EXPAND_SRC_ARGS: u8 = 2;
pub const EXPAND_DST_ARGS: u8 = 3;
pub const EXPAND_SRC_VB: u8 = 4;
pub const EXPAND_DST_VB: u8 = 5;
/// Threads of one group, for both passes.
pub const GROUP: u32 = 64;

/// The rewrite pass. User data: records, the records' stride, whether the
/// count buffer is read, the most source vertices any slot holds, the
/// expanded vertices' capacity, whether vertices are expanded at all, two
/// unused.
///
/// Per record: past the count, an empty draw. Else the clamp's bounds
/// again (2^24 vertices, 2^26 vertex instances), `rects = count / 3`, and a
/// draw of `rects * 6` vertices from a range reserved in the allocator, or
/// an empty one if the source range or the capacity does not hold it.
pub fn rect_rewrite() -> Vec<u8> {
    let mut b = B::new();
    let [records, src_stride, has_count, max_src_verts] = b.user_row(0);
    let [vert_capacity, need_verts, _, _] = b.user_row(1);
    let src = b.s.root_uav_handle(REWRITE_SRC_ARGS);
    let count = b.s.root_uav_handle(REWRITE_COUNT);
    let alloc = b.s.root_uav_handle(REWRITE_ALLOC);
    let dst = b.s.root_uav_handle(REWRITE_DST_ARGS);
    let zero = b.k(0);

    let r = b.compute_id("threadId.i32", 93, 0);
    let past = b.cmp(pred::IUGE, r, records);
    b.ret_if(past);
    let dst_at = b.mulk(r, 16);

    // Past the title's count (read only when there is one): an empty draw.
    let counted = b.cmp(pred::NE, has_count, zero);
    let [dead]: [V; 1] = b
        .if_(
            counted,
            |b| {
                let zero = b.k(0);
                let live = b.load(count, zero, 1)[0];
                vec![b.cmp(pred::IUGE, r, live)]
            },
            |b| vec![b.s.m.bool(false)],
        )
        .try_into()
        .expect("one value");
    b.if_(
        dead,
        |b| {
            let zero = b.k(0);
            b.store(dst, dst_at, &[zero; 4]);
            vec![]
        },
        |_| vec![],
    );
    b.ret_if(dead);

    let src_at = b.bin(Bin::Mul, r, src_stride);
    let rec = b.load(src, src_at, 4);
    let x = b.uminc(rec[0], 1 << 24);
    let one = b.k(1);
    let divisor = b.umax(x, one);
    let most = b.k(1 << 26);
    let most = b.bin(Bin::UDiv, most, divisor);
    let y = b.umin(rec[1], most);
    let z = rec[2];
    let rects = b.udivk(x, 3);
    let src_verts = b.mulk(rects, 3);
    let out_verts = b.mulk(rects, 6);

    let no_rects = b.cmp(pred::EQ, rects, zero);
    let no_instances = b.cmp(pred::EQ, y, zero);
    let start_past = b.cmp(pred::IUGT, z, max_src_verts);
    let room = b.bin(Bin::Sub, max_src_verts, z);
    let too_many = b.cmp(pred::IUGT, src_verts, room);
    let bad = b.bin(Bin::Or, no_rects, no_instances);
    let bad = b.bin(Bin::Or, bad, start_past);
    let bad = b.bin(Bin::Or, bad, too_many);

    let [out_verts, out_start]: [V; 2] = b
        .if_(
            bad,
            |b| vec![b.k(0), b.k(0)],
            |b| {
                let expanded = b.cmp(pred::NE, need_verts, zero);
                b.if_(
                    expanded,
                    |b| {
                        let over = b.cmp(pred::IUGT, out_verts, vert_capacity);
                        b.if_(
                            over,
                            |b| vec![b.k(0), b.k(0)],
                            |b| {
                                let zero = b.k(0);
                                let before = b.atomic_add(alloc, zero, out_verts);
                                let full = b.cmp(pred::IUGE, before, vert_capacity);
                                let left = b.bin(Bin::Sub, vert_capacity, before);
                                let short = b.cmp(pred::IUGT, out_verts, left);
                                let fail = b.bin(Bin::Or, full, short);
                                let verts = b.sel(fail, zero, out_verts);
                                let start = b.sel(fail, zero, before);
                                vec![verts, start]
                            },
                        )
                    },
                    |b| vec![out_verts, b.k(0)],
                )
            },
        )
        .try_into()
        .expect("two values");
    b.store(dst, dst_at, &[out_verts, y, out_start, rec[3]]);
    b.finish()
}

/// The expansion pass, one group per record (a 2D grid of `grid_x` groups
/// per row). User data: records, the title's records' stride, `grid_x`,
/// the source vertices' bias and stride, the expanded vertices' stride,
/// the element count, one unused. The element table holds two dwords per
/// element: its offset, then its kind (0 f32, 1 f16, 2 unorm16, 3 unorm8)
/// and its component count times 256.
pub fn rect_expand() -> Vec<u8> {
    let mut b = B::new();
    let [records, src_stride, grid_x, vb_src_bias] = b.user_row(0);
    let [vb_src_stride, vb_dst_stride, elem_count, _] = b.user_row(1);
    let elems = b.s.root_srv_handle(EXPAND_ELEMS);
    let src_args = b.s.root_uav_handle(EXPAND_SRC_ARGS);
    let dst_args = b.s.root_uav_handle(EXPAND_DST_ARGS);
    let vb = Vb {
        src: b.s.root_uav_handle(EXPAND_SRC_VB),
        dst: b.s.root_uav_handle(EXPAND_DST_VB),
        src_stride: vb_src_stride,
    };
    let zero = b.k(0);

    let gx = b.compute_id("groupId.i32", 94, 0);
    let gy = b.compute_id("groupId.i32", 94, 1);
    let tx = b.compute_id("threadIdInGroup.i32", 95, 0);
    let row = b.bin(Bin::Mul, gy, grid_x);
    let r = b.bin(Bin::Add, row, gx);
    let past = b.cmp(pred::IUGE, r, records);
    b.ret_if(past);
    let dst_at = b.mulk(r, 16);
    let out_verts = b.load(dst_args, dst_at, 1)[0];
    let rects = b.udivk(out_verts, 6);
    let none = b.cmp(pred::EQ, rects, zero);
    b.ret_if(none);
    let at = b.addk(dst_at, 8);
    let dst_start = b.load(dst_args, at, 1)[0];
    let at = b.bin(Bin::Mul, r, src_stride);
    let at = b.addk(at, 8);
    let src_start = b.load(src_args, at, 1)[0];

    let step = b.k(GROUP);
    b.for_(tx, rects, step, vec![], |b, rect, _| {
        let first = b.mulk(rect, 3);
        let first = b.bin(Bin::Add, src_start, first);
        let v0 = b.bin(Bin::Mul, first, vb_src_stride);
        let v0 = b.bin(Bin::Add, vb_src_bias, v0);
        let v1 = b.bin(Bin::Add, v0, vb_src_stride);
        let v2 = b.bin(Bin::Add, v1, vb_src_stride);
        let d = b.mulk(rect, 6);
        let d = b.bin(Bin::Add, dst_start, d);
        let d = b.bin(Bin::Mul, d, vb_dst_stride);

        // The six vertices: v0 v1 v2, v0 v2, and v0 again as the fourth
        // corner's base.
        let (zero, six, one) = (b.k(0), b.k(6), b.k(1));
        b.for_(zero, six, one, vec![], |b, n, _| {
            let is1 = b.cmpk(pred::EQ, n, 1);
            let is2 = b.cmpk(pred::EQ, n, 2);
            let is4 = b.cmpk(pred::EQ, n, 4);
            let is2or4 = b.bin(Bin::Or, is2, is4);
            let from = b.sel(is2or4, v2, v0);
            let from = b.sel(is1, v1, from);
            let to = b.bin(Bin::Mul, vb_dst_stride, n);
            let to = b.bin(Bin::Add, d, to);
            copy_vertex(b, &vb, from, to);
            vec![]
        });

        let five = b.mulk(vb_dst_stride, 5);
        let v3 = b.bin(Bin::Add, d, five);
        let (zero, one) = (b.k(0), b.k(1));
        b.for_(zero, elem_count, one, vec![], |b, e, _| {
            let at = b.mulk(e, 8);
            let desc = b.load(elems, at, 2);
            let offset = desc[0];
            let kind = b.andk(desc[1], 0xff);
            let comps = b.lshrk(desc[1], 8);
            let is_f32 = b.cmpk(pred::EQ, kind, 0);
            let is_u8 = b.cmpk(pred::EQ, kind, 3);
            let (four, two, one_byte) = (b.k(4), b.k(2), b.k(1));
            let bytes = b.sel(is_u8, one_byte, two);
            let bytes = b.sel(is_f32, four, bytes);
            let (zero, one) = (b.k(0), b.k(1));
            b.for_(zero, comps, one, vec![], |b, c, _| {
                let at = b.bin(Bin::Mul, c, bytes);
                let at = b.bin(Bin::Add, offset, at);
                let p0 = b.bin(Bin::Add, v0, at);
                let p1 = b.bin(Bin::Add, v1, at);
                let p2 = b.bin(Bin::Add, v2, at);
                let a = read_comp(b, &vb, kind, p0);
                let m = read_comp(b, &vb, kind, p1);
                let f = read_comp(b, &vb, kind, p2);
                let sum = b.bin(Bin::Add, a, f);
                let v = b.bin(Bin::Sub, sum, m);
                let to = b.bin(Bin::Add, v3, at);
                write_comp(b, &vb, kind, to, v);
                vec![]
            });
            vec![]
        });
        vec![]
    });
    b.finish()
}

/// The expansion's vertex buffers.
struct Vb {
    src: V,
    dst: V,
    src_stride: V,
}

/// Copies one source vertex's `src_stride` bytes to the dword-aligned
/// `to`: whole dwords, then the last one to three bytes in a dword of
/// their own.
fn copy_vertex(b: &mut B, vb: &Vb, from: V, to: V) {
    let full = b.andk(vb.src_stride, !3);
    let (zero, four) = (b.k(0), b.k(4));
    b.for_(zero, full, four, vec![], |b, at, _| {
        let p = b.bin(Bin::Add, from, at);
        let w = load_u32(b, vb, p);
        let q = b.bin(Bin::Add, to, at);
        b.store(vb.dst, q, &[w]);
        vec![]
    });
    let rem = b.andk(vb.src_stride, 3);
    let zero = b.k(0);
    let some = b.cmp(pred::NE, rem, zero);
    b.if_(
        some,
        |b| {
            let base = b.bin(Bin::Add, from, full);
            // Each byte read only if it is part of the vertex, as the HLSL's
            // loop did: a root view has no bounds check.
            let mut w = load_u8(b, vb, base);
            for k in 1..3u32 {
                let more = b.cmpk(pred::IUGT, rem, k);
                let got: [V; 1] = b
                    .if_(
                        more,
                        |b| {
                            let p = b.addk(base, k);
                            let byte = load_u8(b, vb, p);
                            let byte = b.shlk(byte, k * 8);
                            vec![b.bin(Bin::Or, w, byte)]
                        },
                        |_| vec![w],
                    )
                    .try_into()
                    .expect("one value");
                w = got[0];
            }
            let q = b.bin(Bin::Add, to, full);
            b.store(vb.dst, q, &[w]);
            vec![]
        },
        |_| vec![],
    );
}

fn load_u8(b: &mut B, vb: &Vb, pos: V) -> V {
    let aligned = b.andk(pos, !3);
    let w = b.load(vb.src, aligned, 1)[0];
    let rem = b.andk(pos, 3);
    let shift = b.mulk(rem, 8);
    let w = b.lshr(w, shift);
    b.andk(w, 0xff)
}

fn load_u16(b: &mut B, vb: &Vb, pos: V) -> V {
    let rem = b.andk(pos, 3);
    let aligned = b.andk(pos, !3);
    let w0 = b.load(vb.src, aligned, 1)[0];
    let inside = b.cmpk(pred::IULE, rem, 2);
    let [v]: [V; 1] = b
        .if_(
            inside,
            |b| {
                let shift = b.mulk(rem, 8);
                let v = b.lshr(w0, shift);
                vec![b.andk(v, 0xffff)]
            },
            |b| {
                let next = b.addk(aligned, 4);
                let w1 = b.load(vb.src, next, 1)[0];
                let lo = b.lshrk(w0, 24);
                let hi = b.shlk(w1, 8);
                let v = b.bin(Bin::Or, lo, hi);
                vec![b.andk(v, 0xffff)]
            },
        )
        .try_into()
        .expect("one value");
    v
}

fn load_u32(b: &mut B, vb: &Vb, pos: V) -> V {
    let rem = b.andk(pos, 3);
    let aligned = b.andk(pos, !3);
    let w0 = b.load(vb.src, aligned, 1)[0];
    let whole = b.cmpk(pred::EQ, rem, 0);
    let [v]: [V; 1] = b
        .if_(
            whole,
            |_| vec![w0],
            |b| {
                let next = b.addk(aligned, 4);
                let w1 = b.load(vb.src, next, 1)[0];
                let shift = b.mulk(rem, 8);
                let lo = b.lshr(w0, shift);
                let four = b.k(4);
                let inv = b.bin(Bin::Sub, four, rem);
                let inv = b.mulk(inv, 8);
                let hi = b.shl(w1, inv);
                vec![b.bin(Bin::Or, lo, hi)]
            },
        )
        .try_into()
        .expect("one value");
    v
}

/// Replaces `width` bits (8 or 16) at bit `shift` of the dword at
/// `aligned` of the expanded vertices with `val`'s low bits.
fn store_bits(b: &mut B, vb: &Vb, aligned: V, shift: V, val: V, width: u32) {
    let ones = b.k((1u32 << width) - 1);
    let mask = b.shl(ones, shift);
    let keep = b.xork(mask, u32::MAX);
    let w = b.load(vb.dst, aligned, 1)[0];
    let w = b.bin(Bin::And, w, keep);
    let v = b.bin(Bin::And, val, ones);
    let v = b.shl(v, shift);
    let w = b.bin(Bin::Or, w, v);
    b.store(vb.dst, aligned, &[w]);
}

fn store_u8(b: &mut B, vb: &Vb, pos: V, val: V) {
    let aligned = b.andk(pos, !3);
    let rem = b.andk(pos, 3);
    let shift = b.mulk(rem, 8);
    store_bits(b, vb, aligned, shift, val, 8);
}

fn store_u16(b: &mut B, vb: &Vb, pos: V, val: V) {
    let rem = b.andk(pos, 3);
    let inside = b.cmpk(pred::IULE, rem, 2);
    b.if_(
        inside,
        |b| {
            let aligned = b.andk(pos, !3);
            let shift = b.mulk(rem, 8);
            store_bits(b, vb, aligned, shift, val, 16);
            vec![]
        },
        |b| {
            let lo = b.andk(val, 0xff);
            store_u8(b, vb, pos, lo);
            let next = b.addk(pos, 1);
            let hi = b.lshrk(val, 8);
            let hi = b.andk(hi, 0xff);
            store_u8(b, vb, next, hi);
            vec![]
        },
    );
}

fn store_u32(b: &mut B, vb: &Vb, pos: V, val: V) {
    let rem = b.andk(pos, 3);
    let whole = b.cmpk(pred::EQ, rem, 0);
    b.if_(
        whole,
        |b| {
            b.store(vb.dst, pos, &[val]);
            vec![]
        },
        |b| {
            let aligned = b.andk(pos, !3);
            let shift = b.mulk(rem, 8);
            let four = b.k(4);
            let inv = b.bin(Bin::Sub, four, rem);
            let inv = b.mulk(inv, 8);
            let one = b.k(1);
            let mask = b.shl(one, shift);
            let mask = b.subk(mask, 1);
            let w0 = b.load(vb.dst, aligned, 1)[0];
            let w0 = b.bin(Bin::And, w0, mask);
            let lo = b.shl(val, shift);
            let w0 = b.bin(Bin::Or, w0, lo);
            b.store(vb.dst, aligned, &[w0]);
            let next = b.addk(aligned, 4);
            let w1 = b.load(vb.dst, next, 1)[0];
            let keep = b.xork(mask, u32::MAX);
            let w1 = b.bin(Bin::And, w1, keep);
            let hi = b.lshr(val, inv);
            let w1 = b.bin(Bin::Or, w1, hi);
            b.store(vb.dst, next, &[w1]);
            vec![]
        },
    );
}

/// A half's bits as a float, subnormals included.
fn half_to_f32(b: &mut B, h: V) -> V {
    let sgn = b.lshrk(h, 15);
    let sgn = b.shlk(sgn, 31);
    let ex = b.lshrk(h, 10);
    let ex = b.andk(ex, 0x1f);
    let man = b.andk(h, 0x3ff);
    // Zero exponent: zero, or a subnormal (man * 2^-24).
    let f = b.s.t.f32;
    let mf = b.s.m.cast(b.cur, Cast::UiToFp, man, f);
    let scale = b.s.m.f32((1.0f32 / 16_777_216.0).to_bits());
    let v = b.bin(Bin::Mul, mf, scale);
    let neg_zero = b.s.m.f32(0x8000_0000);
    let nv = b.bin(Bin::Sub, neg_zero, v);
    let negative = b.cmpk(pred::NE, sgn, 0);
    let sub = b.sel(negative, nv, v);
    let signed_zero = b.bitcast_f32(sgn);
    let no_man = b.cmpk(pred::EQ, man, 0);
    let small = b.sel(no_man, signed_zero, sub);
    // Infinity and NaN.
    let manh = b.shlk(man, 13);
    let inf = b.ork(sgn, 0x7f80_0000);
    let inf = b.bin(Bin::Or, inf, manh);
    let inf = b.bitcast_f32(inf);
    // Normal.
    let e = b.addk(ex, 112);
    let e = b.shlk(e, 23);
    let n = b.bin(Bin::Or, sgn, e);
    let n = b.bin(Bin::Or, n, manh);
    let normal = b.bitcast_f32(n);
    let is_zero_ex = b.cmpk(pred::EQ, ex, 0);
    let is_max_ex = b.cmpk(pred::EQ, ex, 31);
    let v = b.sel(is_max_ex, inf, normal);
    b.sel(is_zero_ex, small, v)
}

/// A float as a half's bits: to nearest, halfway cases to even,
/// subnormals included, past the range infinity (as `primitive.rs`).
fn f32_to_half(b: &mut B, v: V) -> V {
    let i32t = b.s.t.i32;
    let bits = b.s.m.cast(b.cur, Cast::Bitcast, v, i32t);
    let sgn = b.lshrk(bits, 16);
    let sgn = b.andk(sgn, 0x8000);
    let ex = b.lshrk(bits, 23);
    let ex = b.andk(ex, 0xff);
    let ex = b.subk(ex, 112); // - 127 + 15, signed
    let man = b.andk(bits, 0x7f_ffff);
    // Normal: the rounding carry raises the exponent.
    let retained = b.lshrk(man, 13);
    let retained = b.andk(retained, 1);
    let m = b.addk(man, 0x0fff);
    let m = b.bin(Bin::Add, m, retained);
    let m = b.lshrk(m, 13);
    let e = b.shlk(ex, 10);
    let e = b.bin(Bin::Add, e, m);
    let normal = b.bin(Bin::Or, sgn, e);
    // Subnormal (-10 <= ex <= 0): shift 14..24, so every shift below is
    // in range there; elsewhere it is masked and its result unused.
    let sig = b.ork(man, 1 << 23);
    let fourteen = b.k(14);
    let shift = b.bin(Bin::Sub, fourteen, ex);
    let shift = b.andk(shift, 31);
    let shift1 = b.subk(shift, 1);
    let one = b.k(1);
    let halfway = b.shl(one, shift1);
    let q = b.lshr(sig, shift);
    let low = b.shl(one, shift);
    let low = b.subk(low, 1);
    let r = b.bin(Bin::And, sig, low);
    let above = b.cmp(pred::IUGT, r, halfway);
    let tie = b.cmp(pred::EQ, r, halfway);
    let odd = b.andk(q, 1);
    let odd = b.cmpk(pred::NE, odd, 0);
    let tie_up = b.bin(Bin::And, tie, odd);
    let up = b.bin(Bin::Or, above, tie_up);
    let up = b.s.m.cast(b.cur, Cast::ZExt, up, i32t);
    let qs = b.bin(Bin::Add, q, up);
    let sub = b.bin(Bin::Or, sgn, qs);
    let big = b.cmpk(pred::SGE, ex, 31);
    let inf = b.ork(sgn, 0x7c00);
    let tiny = b.cmpk(pred::SLT, ex, -10i32 as u32);
    let low_range = b.cmpk(pred::SLE, ex, 0);
    let small = b.sel(tiny, sgn, sub);
    let v = b.sel(low_range, small, normal);
    b.sel(big, inf, v)
}

fn read_comp(b: &mut B, vb: &Vb, kind: V, pos: V) -> V {
    let is_f32 = b.cmpk(pred::EQ, kind, 0);
    let [v]: [V; 1] = b
        .if_(
            is_f32,
            |b| {
                let w = load_u32(b, vb, pos);
                vec![b.bitcast_f32(w)]
            },
            |b| {
                let is_f16 = b.cmpk(pred::EQ, kind, 1);
                b.if_(
                    is_f16,
                    |b| {
                        let h = load_u16(b, vb, pos);
                        vec![half_to_f32(b, h)]
                    },
                    |b| {
                        let is_u16 = b.cmpk(pred::EQ, kind, 2);
                        b.if_(
                            is_u16,
                            |b| {
                                let u = load_u16(b, vb, pos);
                                vec![b.unorm(u, 65535.0)]
                            },
                            |b| {
                                let u = load_u8(b, vb, pos);
                                vec![b.unorm(u, 255.0)]
                            },
                        )
                    },
                )
            },
        )
        .try_into()
        .expect("one value");
    v
}

fn write_comp(b: &mut B, vb: &Vb, kind: V, pos: V, v: V) {
    let is_f32 = b.cmpk(pred::EQ, kind, 0);
    b.if_(
        is_f32,
        |b| {
            let i32t = b.s.t.i32;
            let w = b.s.m.cast(b.cur, Cast::Bitcast, v, i32t);
            store_u32(b, vb, pos, w);
            vec![]
        },
        |b| {
            let is_f16 = b.cmpk(pred::EQ, kind, 1);
            b.if_(
                is_f16,
                |b| {
                    let h = f32_to_half(b, v);
                    store_u16(b, vb, pos, h);
                    vec![]
                },
                |b| {
                    let is_u16 = b.cmpk(pred::EQ, kind, 2);
                    b.if_(
                        is_u16,
                        |b| {
                            let u = b.pack_unorm(v, 65535.0);
                            store_u16(b, vb, pos, u);
                            vec![]
                        },
                        |b| {
                            let u = b.pack_unorm(v, 255.0);
                            store_u8(b, vb, pos, u);
                            vec![]
                        },
                    );
                    vec![]
                },
            );
            vec![]
        },
    );
}

/// The writer with a current block, structured control flow and one exit.
struct B {
    s: Shader,
    cur: usize,
    exit: usize,
    user: Option<V>,
}

impl B {
    fn new() -> B {
        let mut s = Shader::new(false);
        s.threads = Some([GROUP, 1, 1]);
        s.user_data = 8;
        let cur = s.entry;
        let exit = s.m.new_block();
        B {
            s,
            cur,
            exit,
            user: None,
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.s.m.terminate(self.cur, Term::Br(self.exit));
        self.s.m.terminate(self.exit, Term::Ret);
        self.s.finish()
    }

    /// Row `row` (four dwords) of the user data.
    fn user_row(&mut self, row: u32) -> [V; 4] {
        let user = match self.user {
            Some(u) => u,
            None => {
                let u = self.s.user_data_handle();
                self.user = Some(u);
                u
            }
        };
        let row = self.k(row);
        let ret = self.s.t.cbuf_ret;
        let v = self.s.op(
            self.cur,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            ret,
            59,
            &[user, row],
        );
        std::array::from_fn(|i| self.s.m.extract(self.cur, v, i as u32))
    }

    fn compute_id(&mut self, name: &str, opcode: u32, component: u32) -> V {
        let i32t = self.s.t.i32;
        let c = self.k(component);
        self.s
            .op(self.cur, name, Attr::ReadNone, i32t, opcode, &[c])
    }

    fn k(&mut self, v: u32) -> V {
        self.s.m.i32(v)
    }

    fn bin(&mut self, op: Bin, a: V, b: V) -> V {
        self.s.m.bin(self.cur, op, a, b)
    }

    fn bink(&mut self, op: Bin, a: V, k: u32) -> V {
        let k = self.k(k);
        self.bin(op, a, k)
    }

    fn addk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Add, a, k)
    }
    fn subk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Sub, a, k)
    }
    fn mulk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Mul, a, k)
    }
    fn udivk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::UDiv, a, k)
    }
    fn andk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::And, a, k)
    }
    fn ork(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Or, a, k)
    }
    fn xork(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Xor, a, k)
    }
    fn shlk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::Shl, a, k)
    }
    fn lshrk(&mut self, a: V, k: u32) -> V {
        self.bink(Bin::LShr, a, k)
    }

    /// Shifts by a computed amount, masked to five bits as HLSL's are.
    fn shl(&mut self, a: V, by: V) -> V {
        let by = self.andk(by, 31);
        self.bin(Bin::Shl, a, by)
    }
    fn lshr(&mut self, a: V, by: V) -> V {
        let by = self.andk(by, 31);
        self.bin(Bin::LShr, a, by)
    }

    fn cmp(&mut self, p: u8, a: V, b: V) -> V {
        self.s.m.cmp(self.cur, p, a, b)
    }
    fn cmpk(&mut self, p: u8, a: V, k: u32) -> V {
        let k = self.k(k);
        self.cmp(p, a, k)
    }
    fn sel(&mut self, c: V, t: V, f: V) -> V {
        self.s.m.select(self.cur, c, t, f)
    }
    fn umin(&mut self, a: V, b: V) -> V {
        let lt = self.cmp(pred::IULT, a, b);
        self.sel(lt, a, b)
    }
    fn umax(&mut self, a: V, b: V) -> V {
        let gt = self.cmp(pred::IUGT, a, b);
        self.sel(gt, a, b)
    }
    fn uminc(&mut self, a: V, most: u32) -> V {
        let most = self.k(most);
        self.umin(a, most)
    }

    fn bitcast_f32(&mut self, v: V) -> V {
        let f = self.s.t.f32;
        self.s.m.cast(self.cur, Cast::Bitcast, v, f)
    }

    /// `u / max` as a float (`max` 255 or 65535).
    fn unorm(&mut self, u: V, max: f32) -> V {
        let f = self.s.t.f32;
        let uf = self.s.m.cast(self.cur, Cast::UiToFp, u, f);
        let max = self.s.m.f32(max.to_bits());
        // LLVM's fdiv shares sdiv's code.
        self.bin(Bin::SDiv, uf, max)
    }

    /// `(uint)(saturate(v) * max + 0.5)`.
    fn pack_unorm(&mut self, v: V, max: f32) -> V {
        let f = self.s.t.f32;
        let sat = self.s.op(self.cur, "unary.f32", Attr::ReadNone, f, 7, &[v]);
        let max = self.s.m.f32(max.to_bits());
        let x = self.bin(Bin::Mul, sat, max);
        let half = self.s.m.f32(0.5f32.to_bits());
        let x = self.bin(Bin::Add, x, half);
        let i32t = self.s.t.i32;
        self.s.m.cast(self.cur, Cast::FpToUi, x, i32t)
    }

    /// `n` dwords (1, 2 or 4) of a raw buffer at byte `at`.
    fn load(&mut self, buffer: V, at: V, n: u32) -> Vec<V> {
        let i32t = self.s.t.i32;
        let ret = self
            .s
            .m
            .named_struct("dx.types.ResRet.i32", vec![i32t, i32t, i32t, i32t, i32t]);
        let undef = self.s.m.undef(i32t);
        let mask = self.s.m.i8(((1u32 << n) - 1) as u8);
        let align = self.k(4);
        let r = self.s.op(
            self.cur,
            "rawBufferLoad.i32",
            Attr::ReadOnly,
            ret,
            139,
            &[buffer, at, undef, mask, align],
        );
        (0..n).map(|i| self.s.m.extract(self.cur, r, i)).collect()
    }

    /// Writes `values` (1 to 4 dwords) to a raw buffer at byte `at`.
    fn store(&mut self, buffer: V, at: V, values: &[V]) {
        let (i32t, void) = (self.s.t.i32, self.s.t.void);
        let undef = self.s.m.undef(i32t);
        let mut v = [undef; 4];
        v[..values.len()].copy_from_slice(values);
        let mask = self.s.m.i8(((1u32 << values.len()) - 1) as u8);
        let align = self.k(4);
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[buffer, at, undef, v[0], v[1], v[2], v[3], mask, align],
        );
    }

    /// Adds `value` to the dword at byte `at` and returns what it held.
    fn atomic_add(&mut self, buffer: V, at: V, value: V) -> V {
        let i32t = self.s.t.i32;
        let undef = self.s.m.undef(i32t);
        let add = self.k(0);
        self.s.op(
            self.cur,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            i32t,
            78,
            &[buffer, add, at, undef, undef, value],
        )
    }

    /// Leaves the shader if `cond` holds.
    fn ret_if(&mut self, cond: V) {
        let next = self.s.m.new_block();
        self.s
            .m
            .terminate(self.cur, Term::CondBr(cond, self.exit, next));
        self.cur = next;
    }

    /// `cond ? then : els`, each arm giving the same number of values of
    /// the same types; the join's phis are returned.
    fn if_(
        &mut self,
        cond: V,
        then: impl FnOnce(&mut B) -> Vec<V>,
        els: impl FnOnce(&mut B) -> Vec<V>,
    ) -> Vec<V> {
        let (t, e, join) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(self.cur, Term::CondBr(cond, t, e));
        self.cur = t;
        let tv = then(self);
        let t_end = self.cur;
        self.s.m.terminate(t_end, Term::Br(join));
        self.cur = e;
        let ev = els(self);
        let e_end = self.cur;
        self.s.m.terminate(e_end, Term::Br(join));
        let arms = [(tv, t_end), (ev, e_end)];
        self.cur = join;
        let n = arms[0].0.len();
        assert_eq!(n, arms[1].0.len(), "both arms give the same values");
        (0..n)
            .map(|i| {
                let ty = self.s.m.type_of(arms[0].0[i]);
                let phi = self.s.m.phi(join, ty);
                self.s
                    .m
                    .set_phi(phi, arms.iter().map(|(v, end)| (v[i], *end)).collect());
                phi
            })
            .collect()
    }

    /// `for (i = start; i < end; i += step)`, unsigned, with values carried
    /// from one iteration to the next; returns their values after the loop.
    fn for_(
        &mut self,
        start: V,
        end: V,
        step: V,
        carried: Vec<V>,
        body: impl FnOnce(&mut B, V, &[V]) -> Vec<V>,
    ) -> Vec<V> {
        let pre = self.cur;
        let (head, first, exit) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s.m.terminate(pre, Term::Br(head));
        let i32t = self.s.t.i32;
        let i = self.s.m.phi(head, i32t);
        let phis: Vec<V> = carried
            .iter()
            .map(|&v| {
                let ty = self.s.m.type_of(v);
                self.s.m.phi(head, ty)
            })
            .collect();
        self.cur = head;
        let more = self.cmp(pred::IULT, i, end);
        self.s.m.terminate(head, Term::CondBr(more, first, exit));
        self.cur = first;
        let next = body(self, i, &phis);
        assert_eq!(
            next.len(),
            carried.len(),
            "the body gives every carried value"
        );
        let stepped = self.bin(Bin::Add, i, step);
        let latch = self.cur;
        self.s.m.terminate(latch, Term::Br(head));
        self.s.m.set_phi(i, vec![(start, pre), (stepped, latch)]);
        for ((&phi, &init), &n) in phis.iter().zip(&carried).zip(&next) {
            self.s.m.set_phi(phi, vec![(init, pre), (n, latch)]);
        }
        self.cur = exit;
        phis
    }
}
