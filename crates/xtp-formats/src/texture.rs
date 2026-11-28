//! Texture layouts of the console GPU (RDNA2, GFX10.3): a texture's size and
//! alignment, where each mip and array slice starts, the address of each
//! element, and copies between tiled memory and linear rows.
//!
//! Units: an element is a texel, or a 4x4 block of a block-compressed
//! format. Mips of one array slice (or one depth slab of a thick 3D texture)
//! are stored together, smallest first: the mip tail (the small mips sharing
//! one block) at offset 0, mip 0 last.
//!
//! The algorithm follows AMD's address library (addrlib), as shipped in Mesa
//! under `src/amd/addrlib`: block dimensions, mip placement and mip tails
//! from `gfx10addrlib.cpp` and `addrlib2.cpp`
//! (`ComputeSurfaceInfoMacroTiled`, `ComputeThinBlockDimension`,
//! `GetMipTailDim`, `GetMaxNumMipsInTail`, `HwlComputeSurfaceInfoLinear`,
//! `ComputeSurfaceAddrFromCoordMacroTiled` with its pipe and bank XOR,
//! `GetSwizzlePatternInfo`, `HwlInitGlobalParams`), and the block patterns
//! of `gfx10SwizzlePattern.h` (the RB+ tables, as on RDNA2), which
//! `texture/generate.rs` turns into `texture/patterns.rs`. addrlib carries
//! this notice:
//!
//! Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//! SPDX-License-Identifier: MIT
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including
//! without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit
//! persons to whom the Software is furnished to do so, subject to the
//! following conditions: the above copyright notice and this permission
//! notice shall be included in all copies or substantial portions of the
//! Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//! EXPRESS OR IMPLIED.

use std::borrow::Cow;
use std::sync::OnceLock;

/// A swizzle mode, by its hardware number (addrlib's `ADDR_SW_*` order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub u32);

impl Swizzle {
    pub const LINEAR: Swizzle = Swizzle(0);
    pub const SW_256B_S: Swizzle = Swizzle(1);
    pub const SW_256B_D: Swizzle = Swizzle(2);
    pub const SW_4KB_S: Swizzle = Swizzle(5);
    pub const SW_4KB_D: Swizzle = Swizzle(6);
    pub const SW_64KB_S: Swizzle = Swizzle(9);
    pub const SW_64KB_D: Swizzle = Swizzle(10);
    pub const SW_64KB_S_T: Swizzle = Swizzle(17);
    pub const SW_64KB_D_T: Swizzle = Swizzle(18);
    pub const SW_4KB_S_X: Swizzle = Swizzle(21);
    pub const SW_4KB_D_X: Swizzle = Swizzle(22);
    pub const SW_4KB_R_X: Swizzle = Swizzle(23);
    pub const SW_64KB_Z_X: Swizzle = Swizzle(24);
    pub const SW_64KB_S_X: Swizzle = Swizzle(25);
    pub const SW_64KB_D_X: Swizzle = Swizzle(26);
    pub const SW_64KB_R_X: Swizzle = Swizzle(27);

    /// log2 of the block size in bytes (0 for linear); `None` for modes
    /// GFX10 does not have.
    pub fn block_log2(self) -> Option<u32> {
        Some(match self.0 {
            0 => 0,
            1 | 2 => 8,
            5 | 6 | 21..=23 => 12,
            9 | 10 | 17 | 18 | 24..=27 => 16,
            _ => return None,
        })
    }

    fn standard(self) -> bool {
        matches!(self.0, 1 | 5 | 9 | 17 | 21 | 25)
    }

    fn display(self) -> bool {
        matches!(self.0, 2 | 6 | 10 | 18 | 22 | 26)
    }

    fn z_order(self) -> bool {
        self.0 == 24
    }

    /// Pipe and bank XOR modes (`_T`, `_X`): their addresses depend on the
    /// GPU's pipe configuration.
    pub fn xor(self) -> bool {
        self.0 >= 16
    }
}

/// The GPU's pipe and packer counts (log2), from its address configuration
/// (`GB_ADDR_CONFIG`): the addresses of the XOR modes depend on them, the
/// others' do not. Pipe interleave is 256 bytes on every console.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pipes {
    pub pipes_log2: u32,
    pub packers_log2: u32,
}

impl Pipes {
    /// Xbox Series X: 32 pipes, 8 packers.
    pub const SERIES_X: Pipes = Pipes {
        pipes_log2: 5,
        packers_log2: 3,
    };
    /// Xbox Series S: 8 pipes, 4 packers.
    pub const SERIES_S: Pipes = Pipes {
        pipes_log2: 3,
        packers_log2: 2,
    };
}

#[rustfmt::skip]
mod patterns;

#[cfg(test)]
mod generate;

/// A block's swizzle pattern (generated in `patterns.rs`): for each address
/// bit inside the block, low to high, the element coordinate bits whose
/// parity it takes ([`x`], [`y`] and [`z`] bits; 0 for the bytes inside an
/// element). Coordinates above the block take part in the XOR modes.
struct Pattern {
    swizzle: Swizzle,
    /// The 3D pattern of a thick block; else z is the array or depth slice.
    thick: bool,
    /// The configuration an XOR mode's pattern is for; `None` for the modes
    /// without XOR, whose patterns hold for every one.
    pipes: Option<Pipes>,
    element_log2: u32,
    bits: &'static [u64],
}

/// Coordinate bits of a pattern's address bit: x in bits 0-15, y 16-31, z
/// 32-47.
const fn x(bit: u32) -> u64 {
    1 << bit
}

const fn y(bit: u32) -> u64 {
    1 << (16 + bit)
}

const fn z(bit: u32) -> u64 {
    1 << (32 + bit)
}

/// Bits of a coordinate: every extent is below 2^14 ([`MAX_EXTENT`]).
const COORD_BITS: usize = 14;

/// For each axis and coordinate bit, the address bits inside a block it
/// flips: a pattern's address bits read the other way round.
type Flips = [[u32; COORD_BITS]; 3];

#[derive(Debug, PartialEq, Eq)]
struct SwizzlePattern {
    flips: Flips,
    element_bytes: u32,
    /// Elements that can be copied together at each aligned column.
    row_run: u32,
    /// Byte offsets of those runs inside one block, shared by all mips and layouts.
    run_offsets: Vec<usize>,
}

fn build_pattern(p: &Pattern) -> SwizzlePattern {
    let mut flips = [[0; COORD_BITS]; 3];
    for (bit, &coords) in p.bits.iter().enumerate() {
        for (axis, f) in flips.iter_mut().enumerate() {
            let mask = (coords >> (16 * axis)) & 0xffff;
            for (c, f) in f.iter_mut().enumerate() {
                if mask >> c & 1 != 0 {
                    *f |= 1 << bit;
                }
            }
        }
    }
    let block_log2 = p.bits.len() as u32;
    let bw = if p.thick {
        let amp = (block_log2 - 10) / 3;
        BLOCK_1K_3D[p.element_log2 as usize][0] << amp
    } else {
        let elements_log2 = block_log2 - p.element_log2;
        1u32 << elements_log2.div_ceil(2)
    };
    // Consecutive low X bits that map 1-to-1 to the address bits right above
    // the element offset (and nowhere else, below the 256-byte XOR region)
    // allow one contiguous copy run at every aligned column.
    let mut run_log2 = 0u32;
    while p.element_log2 + run_log2 < 8
        && (1u32 << run_log2) < bw
        && p.bits[(p.element_log2 + run_log2) as usize] == x(run_log2)
        && flips[0][run_log2 as usize] == (1u32 << (p.element_log2 + run_log2))
    {
        run_log2 += 1;
    }
    let row_run = 1u32 << run_log2;
    let run_offsets = (0..bw)
        .step_by(row_run as usize)
        .map(|col| flip(&flips[0], col) as usize)
        .collect();
    SwizzlePattern {
        flips,
        element_bytes: 1u32 << p.element_log2,
        row_run,
        run_offsets,
    }
}

/// The pattern of a mode, if this module has it: all GFX10 modes but
/// `4KB_R_X` (addrlib has no pattern for it), single-sampled.
fn shared_pattern(
    swizzle: Swizzle,
    thick: bool,
    element_log2: u32,
    pipes: Pipes,
) -> Option<&'static SwizzlePattern> {
    static PATTERNS: [OnceLock<SwizzlePattern>; patterns::PATTERNS.len()] =
        [const { OnceLock::new() }; patterns::PATTERNS.len()];
    let index = patterns::PATTERNS.iter().position(|p| {
        p.swizzle == swizzle
            && p.thick == thick
            && p.element_log2 == element_log2
            && p.pipes.is_none_or(|q| q == pipes)
    })?;
    Some(PATTERNS[index].get_or_init(|| build_pattern(&patterns::PATTERNS[index])))
}

/// The XOR of the address bits a coordinate's set bits flip.
fn flip(flips: &[u32; COORD_BITS], c: u32) -> u32 {
    flips
        .iter()
        .enumerate()
        .filter(|&(bit, _)| c >> bit & 1 != 0)
        .fold(0, |acc, (_, &f)| acc ^ f)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Tex1d,
    Tex2d,
    Tex3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The description makes no texture (zero size, bad element size, a
    /// mode this dimension cannot use, too many mips).
    Invalid,
    /// A valid layout whose addressing is not implemented here.
    Unsupported,
    /// A buffer is smaller than the layout or the rows need.
    TooSmall,
}

/// One plane of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Desc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth: u32,
    pub mips: u32,
    /// 1, 2, 4, 8 or 16.
    pub element_bytes: u32,
    /// Texels per element side: 1, or 4 for block compression.
    pub block: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

/// Elements along one side of mip `mip`, as data and copies see it: texels
/// halve rounding down (to at least 1), then round up to whole elements.
fn dim(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// Elements along one side of mip `mip` for placement (padding, tail,
/// offsets): addrlib halves mip 0's elements rounding up, which can be one
/// more than [`dim`] for odd sizes.
fn padded_dim(texels: u32, mip: u32, block: u32) -> u32 {
    texels.max(1).div_ceil(block).div_ceil(1 << mip)
}

/// Where one mip lives inside a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mip {
    /// Size in elements (depth 1 unless the texture is 3D).
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Padded size in elements: whole blocks, or the tail's share of one.
    pub pitch: u32,
    pub rows: u32,
    pub depth_pitch: u32,
    /// Byte offset of its first block in the slice (0 in the tail).
    pub offset: u64,
    /// In the mip tail: byte offset inside the tail block and the element
    /// position the mip starts at.
    pub tail: Option<Tail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tail {
    pub offset: u32,
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub desc: Desc,
    /// Block dimensions in elements (linear: the pitch alignment, 1, 1).
    pub block: [u32; 3],
    /// 3D blocks span `block[2]` depth slices (GFX10 3D `S` and `D` modes).
    pub thick: bool,
    /// Array slices, depth slabs of `block[2]` slices (thick), or depth
    /// slices (thin and linear 3D).
    pub slices: u32,
    /// Bytes of one slice: its whole mip chain.
    pub slice_size: u64,
    pub size: u64,
    pub alignment: u64,
    /// The first mip in the tail; `mips.len()` without one.
    pub first_tail_mip: u32,
    pub mips: Vec<Mip>,
    /// The pipe configuration the XOR modes' addresses follow.
    pub pipes: Pipes,
    /// The base swizzle (addrlib's `pipeBankXor`) of an XOR mode, 0 for the
    /// others: pipe bits, then bank bits, XORed into every block from
    /// address bit 8.
    pub pipe_bank_xor: u32,
    /// The shared pattern and precomputed row runs; `None` for linear and
    /// layouts without addressing.
    pattern: Option<&'static SwizzlePattern>,
    /// `pipe_bank_xor` as it lands in a block's offset.
    xor: u32,
}

const MAX_MIPS: u32 = 16;

/// The largest width, height and depth (or array size) accepted, in texels.
/// GFX10's image descriptors hold at most 16384 texels per side, so no
/// texture the GPU can use is refused. The bound also keeps every size
/// computed below within `u64` (and its `u32` steps), which unchecked
/// values from a title would otherwise overflow.
const MAX_EXTENT: u32 = 1 << 14;

/// 256-byte micro tile dimensions by log2 element size, 2D and 3D.
const BLOCK_256_2D: [[u32; 2]; 5] = [[16, 16], [16, 8], [8, 8], [8, 4], [4, 4]];
const BLOCK_256_3D: [[u32; 3]; 5] = [[8, 4, 8], [4, 4, 8], [4, 4, 4], [4, 2, 4], [2, 2, 4]];
const BLOCK_1K_3D: [[u32; 3]; 5] = [[16, 8, 8], [8, 8, 8], [8, 8, 4], [8, 4, 4], [4, 4, 4]];

impl Layout {
    /// The layout on an Xbox Series X, with base swizzle 0.
    pub fn new(desc: &Desc) -> Result<Layout, Error> {
        Layout::for_gpu(desc, Pipes::SERIES_X, 0)
    }

    /// The layout on a GPU of `pipes`, whose XOR modes take base swizzle
    /// `pipe_bank_xor` (its bits past the configuration's pipe and bank bits
    /// are ignored, as the GPU does).
    pub fn for_gpu(desc: &Desc, pipes: Pipes, pipe_bank_xor: u32) -> Result<Layout, Error> {
        let d = *desc;
        let block_log2 = d.swizzle.block_log2().ok_or(Error::Invalid)?;
        let valid = (1..=MAX_EXTENT).contains(&d.width)
            && (1..=MAX_EXTENT).contains(&d.height)
            && (1..=MAX_EXTENT).contains(&d.depth)
            && (1..=MAX_MIPS).contains(&d.mips)
            && d.element_bytes.is_power_of_two()
            && d.element_bytes <= 16
            && matches!(d.block, 1 | 4)
            && d.samples.is_power_of_two()
            && d.samples <= 8
            && (d.samples == 1 || (d.dimension == Dimension::Tex2d && d.mips == 1))
            && (d.dimension != Dimension::Tex1d || d.height == 1)
            && (d.dimension != Dimension::Tex3d
                || matches!(d.swizzle.0, 0 | 5 | 9 | 17 | 21 | 24..=27));
        if !valid {
            return Err(Error::Invalid);
        }
        if block_log2 == 0 {
            Ok(Layout::linear(d, pipes))
        } else {
            Ok(Layout::tiled(d, block_log2, pipes, pipe_bank_xor))
        }
    }

    /// `HwlComputeSurfaceInfoLinear`: rows aligned to 256 bytes, mips packed
    /// smallest first, a 3D texture's depth slices each hold the chain.
    fn linear(d: Desc, pipes: Pipes) -> Layout {
        let bpe = u64::from(d.element_bytes);
        let align = 256 / d.element_bytes;
        let three_d = d.dimension == Dimension::Tex3d;
        let mut mips = vec![];
        let mut offset = 0;
        for i in (0..d.mips).rev() {
            let h = padded_dim(d.height, i, d.block);
            let pitch = padded_dim(d.width, i, d.block).next_multiple_of(align);
            mips.push(Mip {
                width: dim(d.width, i, d.block),
                height: dim(d.height, i, d.block),
                depth: if three_d { dim(d.depth, i, 1) } else { 1 },
                pitch,
                rows: h,
                depth_pitch: if three_d { d.depth } else { 1 },
                offset,
                tail: None,
            });
            offset += u64::from(pitch) * u64::from(h) * bpe;
        }
        mips.reverse();
        Layout {
            desc: d,
            block: [align, 1, 1],
            thick: false,
            slices: d.depth,
            slice_size: offset,
            size: offset * u64::from(d.depth),
            alignment: 256,
            first_tail_mip: d.mips,
            mips,
            pipes,
            pipe_bank_xor: 0,
            pattern: None,
            xor: 0,
        }
    }

    fn tiled(d: Desc, block_log2: u32, pipes: Pipes, pipe_bank_xor: u32) -> Layout {
        let element_log2 = d.element_bytes.trailing_zeros();
        let bpe = u64::from(d.element_bytes);
        let block_bytes = 1u64 << block_log2;
        let thick =
            d.dimension == Dimension::Tex3d && (d.swizzle.standard() || d.swizzle.display());
        let block = if thick {
            // ComputeThickBlockDimension
            let amp = (block_log2 - 10) / 3;
            let rest = (block_log2 - 10) % 3;
            let b = BLOCK_1K_3D[element_log2 as usize];
            [
                b[0] << amp,
                b[1] << (amp + rest / 2),
                b[2] << (amp + u32::from(rest != 0)),
            ]
        } else {
            // ComputeThinBlockDimension
            let samples_log2 = d.samples.trailing_zeros();
            let elements_log2 = block_log2 - element_log2 - samples_log2;
            let wide = samples_log2 & 1 == 0 || block_log2 & 1 != 0;
            let w = (elements_log2 + u32::from(wide)) / 2;
            [1 << w, 1 << (elements_log2 - w), 1]
        };
        let slices = d.depth.next_multiple_of(block[2]);
        let three_d = d.dimension == Dimension::Tex3d;
        // Mip depths placed by a thick layout (thin ones store depth as
        // slices).
        let mip_depth = |i: u32| if thick { padded_dim(d.depth, i, 1) } else { 1 };
        let depth_of = |i: u32| if three_d { dim(d.depth, i, 1) } else { 1 };
        let (width, height) = (|i| dim(d.width, i, d.block), |i| dim(d.height, i, d.block));
        let padded = |i| {
            (
                padded_dim(d.width, i, d.block),
                padded_dim(d.height, i, d.block),
            )
        };
        let mut mips = Vec::with_capacity(d.mips as usize);
        let mut first_tail_mip = d.mips;
        let mut slice_size = 0u64;
        if d.mips == 1 {
            mips.push(Mip {
                width: width(0),
                height: height(0),
                depth: depth_of(0),
                pitch: width(0).next_multiple_of(block[0]),
                rows: height(0).next_multiple_of(block[1]),
                depth_pitch: if three_d { slices } else { 1 },
                offset: 0,
                tail: None,
            });
            slice_size =
                u64::from(mips[0].pitch) * u64::from(mips[0].rows) * bpe * u64::from(d.samples);
        } else {
            // GetMipTailDim: the largest mip the tail takes.
            let mut tail_max = block;
            if thick {
                match block_log2 % 3 {
                    0 => tail_max[1] >>= 1,
                    1 => tail_max[0] >>= 1,
                    _ => tail_max[2] >>= 1,
                }
            } else {
                tail_max[0] >>= 1;
            }
            // GetMaxNumMipsInTail; 256-byte blocks have no tail
            // (`ComputeSurfaceInfoMicroTiled`).
            let effective = if thick {
                block_log2 - (block_log2 - 8) / 3
            } else {
                block_log2
            };
            let max_in_tail = match effective {
                8 => 0,
                9..=11 => 1 + (1 << (effective - 9)),
                _ => effective - 4,
            };
            // dsMipmapHtileFix: 8- and 16-bit depth enter the tail as 32-bit.
            let mut fixed = tail_max;
            if d.swizzle.z_order() && element_log2 <= 1 {
                let b = BLOCK_256_2D[element_log2 as usize];
                fixed[0] /= b[0] / BLOCK_256_2D[2][0];
                fixed[1] /= b[1] / BLOCK_256_2D[2][1];
            }
            let mut sizes = vec![];
            for i in 0..d.mips {
                let (w, h) = padded(i);
                if w <= fixed[0] && h <= fixed[1] && d.mips - i <= max_in_tail {
                    first_tail_mip = i;
                    slice_size += block_bytes / u64::from(block[2]);
                    break;
                }
                let pitch = w.next_multiple_of(block[0]);
                let rows = h.next_multiple_of(block[1]);
                let depth = mip_depth(i).next_multiple_of(block[2]);
                let one = u64::from(pitch) * u64::from(rows) * bpe;
                slice_size += one;
                sizes.push(one * u64::from(block[2]));
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if three_d { depth } else { 1 },
                    offset: 0,
                    tail: None,
                });
            }
            // Standard mips from the smallest up, after the tail block.
            let mut offset = if first_tail_mip < d.mips {
                block_bytes
            } else {
                0
            };
            for i in (0..first_tail_mip as usize).rev() {
                mips[i].offset = offset;
                offset += sizes[i];
            }
            let micro = if thick {
                let b = BLOCK_256_3D[element_log2 as usize];
                [b[0], b[1]]
            } else {
                BLOCK_256_2D[element_log2 as usize]
            };
            let (mut pitch, mut rows) = (tail_max[0], tail_max[1]);
            for i in first_tail_mip..d.mips {
                let m = max_in_tail - 1 - (i - first_tail_mip);
                let offset = if m > 6 { 16 << m } else { m << 8 };
                // Address bits 8.. of the offset, odd ones x, even ones y.
                let bits = |from: u32| (0..6).map(|b| ((offset >> (from + 2 * b)) & 1) << b).sum();
                let (mut x, mut y): (u32, u32) = (bits(9), bits(8));
                if block_log2 & 1 != 0 {
                    std::mem::swap(&mut x, &mut y);
                    if element_log2 & 1 != 0 {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                }
                mips.push(Mip {
                    width: width(i),
                    height: height(i),
                    depth: depth_of(i),
                    pitch,
                    rows,
                    depth_pitch: if thick {
                        mip_depth(i).next_multiple_of(BLOCK_256_3D[element_log2 as usize][2])
                    } else {
                        1
                    },
                    offset: 0,
                    tail: Some(Tail {
                        offset,
                        x: x * micro[0],
                        y: y * micro[1],
                    }),
                });
                pitch = (pitch >> 1).max(micro[0]);
                rows = (rows >> 1).max(micro[1]);
            }
        }
        // Thick layouts store each depth slab's chain together.
        let slab = if thick { u64::from(block[2]) } else { 1 };
        let slice_size = slice_size * slab;
        let slices = slices / block[2];
        let pattern = if d.samples == 1 {
            shared_pattern(d.swizzle, thick, element_log2, pipes)
        } else {
            None
        };
        // ComputeSurfaceAddrFromCoordMacroTiled: the pipe bits and the bank
        // bits above them (GetBankXorBits, 2 column bits between), shifted
        // past the 256-byte pipe interleave, inside the block.
        let xor = if d.swizzle.xor() {
            let p = pipes.pipes_log2;
            let banks = block_log2.saturating_sub(8 + p + 2).min(4);
            let mask = ((1 << p) - 1) | (((1 << banks) - 1) << (p + 2));
            ((pipe_bank_xor & mask) << 8) & ((1 << block_log2) - 1)
        } else {
            0
        };
        Layout {
            desc: d,
            block,
            thick,
            slices,
            slice_size,
            size: slice_size * u64::from(slices),
            alignment: block_bytes,
            first_tail_mip,
            mips,
            pipes,
            pipe_bank_xor: if d.swizzle.xor() { pipe_bank_xor } else { 0 },
            pattern,
            xor,
        }
    }

    /// Whether [`Layout::offset`] and the copies work for this layout:
    /// every single-sampled GFX10 mode but `4KB_R_X`, which addrlib has no
    /// pattern for.
    ///
    /// shortcut: no addressing for MSAA; titles render those on the host
    /// GPU. Add the sample patterns (`*_2xaa` to `*_8xaa`) when a title
    /// copies one through the CPU.
    pub fn addressable(&self) -> bool {
        self.desc.swizzle == Swizzle::LINEAR || self.pattern.is_some()
    }

    /// The slice and the depth inside it for `z` (an array slice or a 3D
    /// depth slice).
    fn split(&self, z: u32) -> (u32, u32) {
        if self.thick {
            (z / self.block[2], z % self.block[2])
        } else {
            (z, 0)
        }
    }

    /// Byte offset of element (`x`, `y`) of mip `mip` in array slice or
    /// depth slice `z`. Coordinates must lie inside the mip.
    pub fn offset(&self, mip: u32, x: u32, y: u32, z: u32) -> Result<u64, Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = if self.desc.dimension == Dimension::Tex3d {
            m.depth
        } else {
            self.slices
        };
        let inside = x < m.width && y < m.height && z < depth;
        if !inside || !self.addressable() {
            return Err(if inside {
                Error::Unsupported
            } else {
                Error::Invalid
            });
        }
        let (slice, _) = self.split(z);
        let base = u64::from(slice) * self.slice_size + m.offset;
        let Some(pattern) = self.pattern else {
            let row = u64::from(y) * u64::from(m.pitch) + u64::from(x);
            return Ok(base + row * u64::from(self.desc.element_bytes));
        };
        let (row, xor) = self.row(&pattern.flips, m, y, z);
        Ok(base + row + (self.column(&pattern.flips, m, x) ^ u64::from(xor)))
    }

    /// The part of an address that depends on the row and the slice: whole
    /// blocks, and the bits it XORs into the block offset (y, slice and the
    /// base swizzle). `z` is the array slice or depth slice: thick blocks
    /// take its bits inside the block, the XOR modes its bits above too.
    fn row(&self, flips: &Flips, m: &Mip, y: u32, z: u32) -> (u64, u32) {
        let [bw, bh, _] = self.block;
        let log2 = self.alignment.trailing_zeros();
        // In the tail the pattern of the shifted position gives the mip's
        // offset inside the tail block.
        let y = y + m.tail.map_or(0, |t| t.y);
        let blocks =
            u64::from(y >> bh.trailing_zeros()) * u64::from(m.pitch >> bw.trailing_zeros());
        let xor = flip(&flips[1], y) ^ flip(&flips[2], z) ^ self.xor;
        (blocks << log2, xor)
    }

    /// The part of an address that depends on the column: whole blocks plus
    /// the bits the column flips inside the block. A row's XOR applies to
    /// the low bits alone, so it combines with this by XOR.
    fn column(&self, flips: &Flips, m: &Mip, x: u32) -> u64 {
        let bw = self.block[0];
        let x = x + m.tail.map_or(0, |t| t.x);
        (u64::from(x / bw) << self.alignment.trailing_zeros()) | u64::from(flip(&flips[0], x))
    }

    /// Depth slices a subresource of mip `mip` covers: the mip's depth for
    /// 3D textures, else 1 (one array slice).
    pub fn subresource_depth(&self, mip: u32) -> u32 {
        self.mips.get(mip as usize).map_or(0, |m| m.depth)
    }

    /// Bytes from the start of the texture that hold `first_mip` and all
    /// smaller mips, including the whole tail block if present. Returns
    /// [`Layout::size`] for mip 0 and for layouts other than single-slice,
    /// single-sampled 2D textures. Returns `None` for an invalid mip index.
    pub fn resident_bytes(&self, first_mip: u32) -> Option<u64> {
        let mip = self.mips.get(first_mip as usize)?;
        if first_mip == 0
            || self.desc.dimension != Dimension::Tex2d
            || self.slices != 1
            || self.desc.samples != 1
        {
            return Some(self.size);
        }
        if mip.tail.is_some() {
            return Some(self.alignment);
        }
        // The next larger mip starts right after this one, including its padding.
        Some(self.mips[first_mip as usize - 1].offset)
    }

    /// Copies mip `mip` of array slice `slice` into linear rows (3D: every
    /// depth slice of the mip, `slice` 0). Rows are `row_pitch` bytes apart,
    /// depth slices `depth_pitch`.
    ///
    /// `tiled` starts at offset 0 of the texture and must hold at least
    /// [`Layout::resident_bytes`] for `mip`, or the copy returns [`Error::TooSmall`].
    pub fn detile(
        &self,
        tiled: &[u8],
        mip: u32,
        slice: u32,
        linear: &mut [u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        let needed = self.resident_bytes(mip).ok_or(Error::Invalid)?;
        self.copy::<true>(tiled, linear, mip, slice, row_pitch, depth_pitch, needed)
    }

    /// Copies linear rows into mip `mip` of array slice `slice`, with the
    /// same row and depth pitches as [`Layout::detile`]. `tiled` must hold
    /// the whole texture ([`Layout::size`] bytes).
    pub fn tile(
        &self,
        tiled: &mut [u8],
        mip: u32,
        slice: u32,
        linear: &[u8],
        row_pitch: usize,
        depth_pitch: usize,
    ) -> Result<(), Error> {
        self.copy::<false>(linear, tiled, mip, slice, row_pitch, depth_pitch, self.size)
    }

    /// Copies contiguous runs between tiled and linear memory.
    #[allow(clippy::too_many_arguments)]
    fn copy<const DETILE: bool>(
        &self,
        src: &[u8],
        dst: &mut [u8],
        mip: u32,
        slice: u32,
        row_pitch: usize,
        depth_pitch: usize,
        tiled_need: u64,
    ) -> Result<(), Error> {
        let m = self.mips.get(mip as usize).ok_or(Error::Invalid)?;
        let depth = m.depth;
        let first = if self.desc.dimension == Dimension::Tex3d {
            if slice != 0 {
                return Err(Error::Invalid);
            }
            0
        } else {
            slice
        };
        if first >= self.slices {
            return Err(Error::Invalid);
        }
        if !self.addressable() {
            return Err(Error::Unsupported);
        }
        let n = self.desc.element_bytes as usize;
        let bytes = m.width as usize * n;
        let need = (depth as usize - 1)
            .checked_mul(depth_pitch)
            .zip((m.height as usize - 1).checked_mul(row_pitch))
            .and_then(|(depth_offset, row_offset)| depth_offset.checked_add(row_offset))
            .and_then(|offset| offset.checked_add(bytes))
            .ok_or(Error::TooSmall)?;
        let (tiled_len, linear_len) = if DETILE {
            (src.len(), dst.len())
        } else {
            (dst.len(), src.len())
        };
        if (tiled_len as u64) < tiled_need || linear_len < need || row_pitch < bytes {
            return Err(Error::TooSmall);
        }
        let regular = self.row_runs(m);
        // Fallback for irregular or unaligned views: compute individual column runs.
        let mut runs = Vec::new();
        if let Some(pattern) = self.pattern
            && regular.is_none()
        {
            let flips = &pattern.flips;
            let mut first_linear = 0;
            let mut first_tiled = self.column(flips, m, 0);
            for x in 1..m.width {
                let offset = self.column(flips, m, x);
                let at = x as usize * n;
                if offset != first_tiled + (at - first_linear) as u64 {
                    runs.push((first_tiled, first_linear, at - first_linear));
                    first_tiled = offset;
                    first_linear = at;
                }
            }
            runs.push((first_tiled, first_linear, bytes - first_linear));
        }
        for z in 0..depth {
            let (s, _) = self.split(first + z);
            let base = u64::from(s) * self.slice_size + m.offset;
            for y in 0..m.height {
                let linear_row = z as usize * depth_pitch + y as usize * row_pitch;
                let Some(pattern) = self.pattern else {
                    let tiled_row = base + u64::from(y) * u64::from(m.pitch) * n as u64;
                    copy_run::<DETILE>(src, dst, tiled_row as usize, linear_row, bytes);
                    continue;
                };
                let (row, xor) = self.row(&pattern.flips, m, y, first + z);
                let tiled_row = (base + row) as usize;
                if let Some((run_bytes, offsets)) = &regular {
                    let x = xor as usize;
                    let (t, l, b, o) = (tiled_row, linear_row, bytes, offsets.as_ref());
                    match *run_bytes {
                        2 => regular_row::<2, DETILE>(src, dst, t, x, l, b, o),
                        4 => regular_row::<4, DETILE>(src, dst, t, x, l, b, o),
                        8 => regular_row::<8, DETILE>(src, dst, t, x, l, b, o),
                        16 => regular_row::<16, DETILE>(src, dst, t, x, l, b, o),
                        32 => regular_row::<32, DETILE>(src, dst, t, x, l, b, o),
                        _ => unreachable!(),
                    }
                    continue;
                }
                for &(tiled, linear, len) in &runs {
                    let at = tiled_row + (tiled ^ u64::from(xor)) as usize;
                    copy_run::<DETILE>(src, dst, at, linear_row + linear, len);
                }
            }
        }
        Ok(())
    }

    /// Borrow the run offsets for a row inside one block (including shifted
    /// mip tails). Wider rows compute one offset per run, once per mip.
    fn row_runs(&self, mip: &Mip) -> Option<(usize, Cow<'static, [usize]>)> {
        let p = self.pattern?;
        let start_x = mip.tail.map_or(0, |t| t.x);
        let run = p.row_run;
        let bytes = (run * p.element_bytes) as usize;
        let bw = self.block[0];
        if !matches!(bytes, 2 | 4 | 8 | 16 | 32)
            || !start_x.is_multiple_of(run)
            || self.desc.element_bytes != p.element_bytes
            || bw as usize != p.run_offsets.len() * run as usize
            || !self.alignment.is_power_of_two()
            || self.alignment < bytes as u64
        {
            return None;
        }
        let first = (start_x / run) as usize;
        let count = mip.width.div_ceil(run) as usize;
        if let Some(slice) = p.run_offsets.get(first..first + count) {
            return Some((bytes, Cow::Borrowed(slice)));
        }
        let per_tile = p.run_offsets.len();
        let shift = per_tile.trailing_zeros();
        let tile_bytes = self.alignment as usize;
        let xor = self.desc.swizzle.xor();
        let offsets = (first..first + count)
            .map(|i| {
                let tile = i >> shift;
                let high = if xor {
                    flip(&p.flips[0], tile as u32 * bw) as usize
                } else {
                    0
                };
                tile * tile_bytes + (p.run_offsets[i & (per_tile - 1)] ^ high)
            })
            .collect();
        Some((bytes, Cow::Owned(offsets)))
    }
}

/// Copies one tiled row using fixed-size chunks. The last run of a mip may be
/// short, and pitched linear rows can start at any byte offset.
fn regular_row<const N: usize, const DETILE: bool>(
    src: &[u8],
    dst: &mut [u8],
    tiled: usize,
    xor: usize,
    linear: usize,
    bytes: usize,
    offsets: &[usize],
) {
    let (count, tail) = (bytes / N, bytes % N);
    if DETILE {
        let chunks = src[tiled..].as_chunks::<N>().0;
        let rows = dst[linear..][..bytes].as_chunks_mut::<N>().0;
        for (out, &offset) in rows.iter_mut().zip(&offsets[..count]) {
            *out = chunks[(offset ^ xor) / N];
        }
    } else {
        let chunks = dst[tiled..].as_chunks_mut::<N>().0;
        let rows = src[linear..][..bytes].as_chunks::<N>().0;
        for (input, &offset) in rows.iter().zip(&offsets[..count]) {
            chunks[(offset ^ xor) / N] = *input;
        }
    }
    if tail != 0 {
        copy_run::<DETILE>(
            src,
            dst,
            tiled + (offsets[count] ^ xor),
            linear + count * N,
            tail,
        );
    }
}

fn copy_run<const DETILE: bool>(
    src: &[u8],
    dst: &mut [u8],
    tiled: usize,
    linear: usize,
    len: usize,
) {
    if DETILE {
        dst[linear..][..len].copy_from_slice(&src[tiled..][..len]);
    } else {
        dst[tiled..][..len].copy_from_slice(&src[linear..][..len]);
    }
}

/// What a DXGI format stores per element: bytes, texels per element side
/// (4 for block compression), and whether its stencil is a second plane
/// (the GPU keeps depth and stencil apart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub bytes: u32,
    pub block: u32,
    pub stencil: bool,
}

/// By DXGI format number. `None` for 96-bit, 1-bit, packed 4:2:2 and video
/// formats.
///
/// shortcut: those formats have no layout here; add them when a title uses
/// one.
pub fn format(dxgi: u32) -> Option<Format> {
    let (bytes, block, stencil) = match dxgi {
        1..=4 => (16, 1, false),
        9..=18 => (8, 1, false),
        19..=22 | 44..=47 => (4, 1, true),
        23..=43 | 67 | 87..=93 | 116 | 117 => (4, 1, false),
        48..=59 | 85 | 86 | 115 => (2, 1, false),
        118..=120 => (2, 1, true),
        60..=65 => (1, 1, false),
        70..=72 | 79..=81 => (8, 4, false),
        73..=78 | 82..=84 | 94..=99 => (16, 4, false),
        _ => return None,
    };
    Some(Format {
        bytes,
        block,
        stencil,
    })
}

/// A texture as the console allocates it: a D3D12-style description with
/// a swizzle mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub dimension: Dimension,
    /// In texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture, else the array size.
    pub depth_or_array: u32,
    /// 0: the full chain.
    pub mips: u32,
    /// DXGI format number.
    pub format: u32,
    pub samples: u32,
    pub swizzle: Swizzle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// From the start of the texture.
    pub offset: u64,
    pub stencil: bool,
    pub layout: Layout,
}

/// Every data plane of a texture (depth, then stencil), placed one after
/// the other.
///
/// shortcut: no compression metadata planes (HTILE, CMASK, FMASK, DCC);
/// the console adds them to render targets and depth buffers unless the
/// description denies them, so those are smaller here than on the console.
/// The host GPU renders them, so nothing reads that memory; add the planes
/// if a title sizes a pool from console sizes and runs out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    pub planes: Vec<Plane>,
    pub size: u64,
    pub alignment: u64,
}

impl Texture {
    /// The texture on an Xbox Series X, with base swizzle 0.
    pub fn new(desc: &TextureDesc) -> Result<Texture, Error> {
        Texture::for_gpu(desc, Pipes::SERIES_X, 0)
    }

    /// The texture on a GPU of `pipes`; every plane takes base swizzle
    /// `pipe_bank_xor` (see [`Layout::for_gpu`]).
    pub fn for_gpu(desc: &TextureDesc, pipes: Pipes, pipe_bank_xor: u32) -> Result<Texture, Error> {
        let f = format(desc.format).ok_or(Error::Invalid)?;
        let depth3d = if desc.dimension == Dimension::Tex3d {
            desc.depth_or_array
        } else {
            1
        };
        let mips = match desc.mips {
            0 => {
                32 - desc
                    .width
                    .max(desc.height)
                    .max(depth3d)
                    .max(1)
                    .leading_zeros()
            }
            n => n,
        };
        let plane = |bytes| Desc {
            dimension: desc.dimension,
            width: desc.width,
            height: desc.height,
            depth: desc.depth_or_array,
            mips,
            element_bytes: bytes,
            block: f.block,
            samples: desc.samples.max(1),
            swizzle: desc.swizzle,
        };
        let layout = |bytes| Layout::for_gpu(&plane(bytes), pipes, pipe_bank_xor);
        let mut planes = vec![Plane {
            offset: 0,
            stencil: false,
            layout: layout(f.bytes)?,
        }];
        if f.stencil {
            let layout = layout(1)?;
            let offset = planes[0].layout.size.next_multiple_of(layout.alignment);
            planes.push(Plane {
                offset,
                stencil: true,
                layout,
            });
        }
        let last = &planes[planes.len() - 1];
        let size = last.offset + last.layout.size;
        let alignment = planes.iter().map(|p| p.layout.alignment).max().unwrap_or(1);
        Ok(Texture {
            planes,
            size,
            alignment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: u32, h: u32, mips: u32, bpe: u32, swizzle: Swizzle) -> Desc {
        Desc {
            dimension: Dimension::Tex2d,
            width: w,
            height: h,
            depth: 1,
            mips,
            element_bytes: bpe,
            block: 1,
            samples: 1,
            swizzle,
        }
    }

    /// Sizes past the hardware's 16384 texels per side are refused instead
    /// of overflowing the size arithmetic (found by random-input testing).
    #[test]
    fn extents_past_the_hardware_limit_are_invalid() {
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_64KB_D_X] {
            let largest = Layout::new(&desc(MAX_EXTENT, MAX_EXTENT, 15, 16, swizzle)).unwrap();
            assert!(largest.size >= u64::from(MAX_EXTENT) * u64::from(MAX_EXTENT) * 16);
            for (w, h) in [
                (MAX_EXTENT + 1, 1),
                (1, MAX_EXTENT + 1),
                (u32::MAX, u32::MAX),
            ] {
                let result = Layout::new(&desc(w, h, 1, 16, swizzle));
                assert_eq!(result, Err(Error::Invalid), "{w}x{h}");
            }
            let mut deep = desc(1, 1, 1, 16, swizzle);
            deep.depth = u32::MAX;
            assert_eq!(Layout::new(&deep), Err(Error::Invalid));
        }
    }

    #[test]
    fn blocks_hold_their_size_in_elements() {
        for log2 in [8u32, 12, 16] {
            let swizzle = match log2 {
                8 => Swizzle::SW_256B_S,
                12 => Swizzle::SW_4KB_S,
                _ => Swizzle::SW_64KB_D,
            };
            for e in 0..5 {
                let l = Layout::new(&desc(1, 1, 1, 1 << e, swizzle)).unwrap();
                let [w, h, d] = l.block;
                assert_eq!(u64::from(w * h * d) << e, 1 << log2, "{log2} {e}");
                // The pattern covers exactly the block's coordinates.
                let flips = l.pattern.unwrap().flips;
                for (axis, side) in [w, h].into_iter().enumerate() {
                    let inside = side.trailing_zeros() as usize;
                    assert!(flips[axis][..inside].iter().all(|&f| f != 0));
                    assert!(flips[axis][inside..].iter().all(|&f| f == 0));
                }
            }
        }
        // 64 KiB, 16-byte elements: 64x64 (a 256x256 BC7 texture).
        let l = Layout::new(&desc(64, 64, 1, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.block, [64, 64, 1]);
        assert_eq!((l.size, l.alignment), (65536, 65536));
    }

    /// Every element of every mip and slice lands on its own address, inside
    /// the texture, and tiling then detiling gives the rows back.
    #[test]
    fn addresses_are_a_bijection_and_copies_round_trip() {
        let cases = [
            desc(64, 64, 7, 16, Swizzle::SW_64KB_D),
            desc(100, 60, 7, 4, Swizzle::SW_64KB_D),
            desc(250, 175, 8, 8, Swizzle::SW_4KB_S),
            desc(33, 17, 6, 1, Swizzle::SW_4KB_D),
            desc(40, 9, 3, 2, Swizzle::SW_64KB_S),
            desc(30, 20, 5, 4, Swizzle::LINEAR),
            desc(83, 31, 6, 1, Swizzle::SW_256B_S),
            Desc {
                block: 4,
                ..desc(109, 71, 6, 16, Swizzle::SW_256B_D)
            },
            Desc {
                depth: 3,
                ..desc(70, 70, 4, 16, Swizzle::SW_4KB_S)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 16,
                ..desc(64, 32, 7, 4, Swizzle::SW_4KB_S)
            },
            // The XOR modes, below with each console's pipes.
            desc(300, 200, 9, 4, Swizzle::SW_64KB_R_X),
            desc(129, 257, 9, 1, Swizzle::SW_64KB_R_X),
            desc(96, 80, 7, 16, Swizzle::SW_64KB_S_X),
            desc(70, 300, 9, 8, Swizzle::SW_64KB_D_X),
            desc(260, 66, 9, 2, Swizzle::SW_64KB_Z_X),
            desc(65, 33, 7, 1, Swizzle::SW_64KB_Z_X),
            desc(90, 50, 7, 2, Swizzle::SW_64KB_S_T),
            desc(77, 99, 7, 16, Swizzle::SW_64KB_D_T),
            desc(61, 45, 6, 4, Swizzle::SW_4KB_S_X),
            desc(50, 70, 7, 8, Swizzle::SW_4KB_D_X),
            Desc {
                block: 4,
                ..desc(200, 120, 6, 16, Swizzle::SW_64KB_S_X)
            },
            Desc {
                depth: 5,
                ..desc(130, 70, 3, 4, Swizzle::SW_64KB_R_X)
            },
            Desc {
                depth: 3,
                ..desc(40, 24, 4, 2, Swizzle::SW_4KB_D_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 20,
                ..desc(40, 36, 6, 4, Swizzle::SW_64KB_S_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 9,
                ..desc(33, 20, 5, 8, Swizzle::SW_64KB_D_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 12,
                ..desc(30, 30, 5, 1, Swizzle::SW_4KB_S_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 7,
                ..desc(24, 40, 5, 16, Swizzle::SW_64KB_S_T)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 6,
                ..desc(70, 40, 4, 4, Swizzle::SW_64KB_Z_X)
            },
        ];
        let gpus = [
            (Pipes::SERIES_X, 0),
            (Pipes::SERIES_S, 0x2d),
            (Pipes::SERIES_X, 0x1f3),
        ];
        let layouts = cases.iter().flat_map(|d| {
            let gpus = if d.swizzle.xor() {
                &gpus[..]
            } else {
                &gpus[..1]
            };
            gpus.iter()
                .map(|&(pipes, xor)| (*d, Layout::for_gpu(d, pipes, xor).unwrap()))
        });
        for (d, l) in layouts {
            assert!(l.addressable(), "{d:?}");
            let mut seen = vec![false; l.size as usize];
            let mut tiled = vec![0u8; l.size as usize];
            let n = d.element_bytes as usize;
            let slices = if d.dimension == Dimension::Tex3d {
                1
            } else {
                d.depth
            };
            for slice in 0..slices {
                for mip in 0..d.mips {
                    let m = l.mips[mip as usize];
                    let active = m.width as usize * n;
                    let row = active + 13;
                    let plane = row * m.height as usize + 11;
                    let linear: Vec<u8> = (0..plane * m.depth as usize)
                        .map(|i| (i * 7 + mip as usize * 13 + slice as usize) as u8)
                        .collect();
                    l.tile(&mut tiled, mip, slice, &linear, row, plane).unwrap();
                    let mut back = vec![0xcc; linear.len()];
                    l.detile(&tiled, mip, slice, &mut back, row, plane).unwrap();
                    for z in 0..m.depth {
                        let base = z as usize * plane;
                        for y in 0..m.height {
                            let start = base + y as usize * row;
                            assert_eq!(back[start..start + active], linear[start..start + active]);
                            assert!(back[start + active..start + row].iter().all(|&b| b == 0xcc));
                            for x in 0..m.width {
                                let at = l.offset(mip, x, y, slice + z).unwrap() as usize;
                                let source = start + x as usize * n;
                                assert_eq!(tiled[at..at + n], linear[source..source + n]);
                                assert!(at + n <= seen.len(), "{d:?} mip {mip} outside");
                                assert!(at.is_multiple_of(n));
                                assert!(!seen[at], "{d:?} mip {mip} ({x},{y},{z}) twice");
                                seen[at] = true;
                            }
                        }
                        assert!(
                            back[base + plane - 11..base + plane]
                                .iter()
                                .all(|&b| b == 0xcc)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn detiles_smaller_mips_from_a_prefix() {
        let cases = [
            desc(250, 175, 8, 8, Swizzle::SW_4KB_S),
            Desc {
                block: 4,
                ..desc(1024, 768, 10, 16, Swizzle::SW_64KB_D)
            },
            desc(30, 20, 5, 4, Swizzle::LINEAR),
            desc(300, 200, 9, 4, Swizzle::SW_64KB_R_X),
        ];
        for d in cases {
            let l = Layout::for_gpu(&d, Pipes::SERIES_S, 0x2d).unwrap();
            let n = d.element_bytes as usize;
            let mut tiled = vec![0; l.size as usize];
            let mut round_trip = vec![0; tiled.len()];
            let mut images = Vec::new();
            for (mip, m) in l.mips.iter().enumerate() {
                let row = m.width as usize * n;
                let image: Vec<u8> = (0..row * m.height as usize)
                    .map(|i| ((i * 7) ^ (i >> 8) ^ (mip * 53)) as u8)
                    .collect();
                for y in 0..m.height {
                    for x in 0..m.width {
                        let at = l.offset(mip as u32, x, y, 0).unwrap() as usize;
                        let from = y as usize * row + x as usize * n;
                        tiled[at..at + n].copy_from_slice(&image[from..from + n]);
                    }
                }
                l.tile(&mut round_trip, mip as u32, 0, &image, row, 0)
                    .unwrap();
                // Size the prefix by the padded extent of this mip or tail.
                let end = if m.tail.is_some() {
                    l.alignment as usize
                } else {
                    m.offset as usize + m.pitch as usize * m.rows as usize * n
                };
                images.push((end, row, image));
            }
            assert_eq!(round_trip, tiled, "{d:?}");
            assert_eq!(images[0].0, tiled.len(), "{d:?}");
            assert!(images[1].0 < tiled.len(), "{d:?}");
            if d.swizzle != Swizzle::LINEAR {
                assert!((2..d.mips - 1).contains(&l.first_tail_mip), "{d:?}");
            }

            for (first, (end, row, image)) in images.iter().enumerate() {
                assert_eq!(l.resident_bytes(first as u32), Some(*end as u64), "{d:?}");
                let mut out = vec![0xcc; image.len()];
                assert_eq!(
                    l.detile(&tiled[..end - 1], first as u32, 0, &mut out, *row, 0),
                    Err(Error::TooSmall),
                    "{d:?} mip {first}"
                );
                assert!(out.iter().all(|&byte| byte == 0xcc));
                for (mip, (needed, row, image)) in images.iter().enumerate() {
                    let mut out = vec![0xcc; image.len()];
                    let result = l.detile(&tiled[..*end], mip as u32, 0, &mut out, *row, 0);
                    if needed <= end {
                        result.unwrap();
                        assert_eq!(&out, image, "{d:?} first mip {first}, mip {mip}");
                    } else {
                        assert_eq!(result, Err(Error::TooSmall), "{d:?} mip {mip}");
                        assert!(out.iter().all(|&byte| byte == 0xcc));
                    }
                }
            }
            assert_eq!(l.resident_bytes(d.mips), None);
            assert_eq!(
                l.detile(&tiled, d.mips, 0, &mut [], 0, 0),
                Err(Error::Invalid)
            );
            let (end, row, image) = &images[1];
            let mut short = vec![0xcc; *end];
            assert_eq!(
                l.tile(&mut short, 1, 0, image, *row, 0),
                Err(Error::TooSmall)
            );
            assert!(short.iter().all(|&byte| byte == 0xcc));
        }
    }

    #[test]
    fn other_layouts_require_the_whole_texture() {
        let base = desc(64, 64, 5, 4, Swizzle::SW_4KB_S);
        for d in [
            Desc { depth: 2, ..base },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 2,
                ..base
            },
            Desc {
                dimension: Dimension::Tex1d,
                ..desc(64, 1, 5, 4, Swizzle::LINEAR)
            },
            Desc {
                samples: 2,
                ..desc(4, 4, 1, 4, Swizzle::LINEAR)
            },
        ] {
            let l = Layout::new(&d).unwrap();
            let mip = d.mips - 1;
            let m = &l.mips[mip as usize];
            let row = (m.width * d.element_bytes) as usize;
            let plane = row * m.height as usize;
            let mut out = vec![0; plane * m.depth as usize];
            let tiled = vec![0; l.size as usize];
            assert_eq!(l.resident_bytes(mip), Some(l.size), "{d:?}");
            assert_eq!(
                l.detile(&tiled[..tiled.len() - 1], mip, 0, &mut out, row, plane),
                Err(Error::TooSmall),
                "{d:?}"
            );
            l.detile(&tiled, mip, 0, &mut out, row, plane).unwrap();
        }
        let msaa = Layout::new(&Desc {
            samples: 2,
            ..desc(4, 4, 1, 4, Swizzle::SW_4KB_S)
        })
        .unwrap();
        assert_eq!(msaa.resident_bytes(0), Some(msaa.size));
        assert_eq!(
            msaa.detile(&vec![0; msaa.size as usize], 0, 0, &mut [0; 64], 16, 0),
            Err(Error::Unsupported)
        );
    }

    /// Hand-checked: in a 64 KiB `D` block of 4-byte elements, x bits go to
    /// address bits 2, 3, 6, 9, 11, 13, 15 and y bits to 4, 5, 7, 8, 10, 12,
    /// 14; a 4 KiB `S` block of 16-byte elements is 16x16 with y first.
    #[test]
    fn hand_checked_addresses() {
        let l = Layout::new(&desc(128, 128, 1, 4, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 2));
        assert_eq!(l.offset(0, 4, 0, 0), Ok(1 << 6));
        assert_eq!(l.offset(0, 0, 4, 0), Ok(1 << 7));
        assert_eq!(l.offset(0, 0, 8, 0), Ok(1 << 8));
        assert_eq!(l.offset(0, 64, 64, 0), Ok((1 << 15) | (1 << 14)));
        let l = Layout::new(&desc(32, 32, 1, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.block, [16, 16, 1]);
        assert_eq!(l.offset(0, 0, 1, 0), Ok(1 << 4));
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 6));
        // Blocks row by row: (16, 0) is the second block, (0, 16) the third.
        assert_eq!(l.offset(0, 16, 0, 0), Ok(4096));
        assert_eq!(l.offset(0, 0, 16, 0), Ok(8192));
    }

    /// A 256x256 BC7 chain in 64 KiB `D`: one tail block holding mips 1-8,
    /// then mip 0; mip 1 sits at x = 32 elements, 32 KiB into the tail.
    #[test]
    fn mip_tail_comes_first() {
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_64KB_D)).unwrap();
        assert_eq!(l.first_tail_mip, 1);
        assert_eq!(l.size, 2 * 65536);
        assert_eq!(l.mips[0].offset, 65536);
        assert_eq!(
            l.mips[1].tail,
            Some(Tail {
                offset: 32768,
                x: 32,
                y: 0
            })
        );
        assert_eq!(l.offset(1, 0, 0, 0), Ok(32768));
        // 4 KiB blocks hold fewer tail mips: 16x16 BC7 mips stay outside.
        let l = Layout::new(&desc(64, 64, 9, 16, Swizzle::SW_4KB_S)).unwrap();
        assert_eq!(l.first_tail_mip, 3);
        assert_eq!(l.mips[2].offset, 4096);
        assert_eq!(l.mips[1].offset, 8192);
        assert_eq!(l.mips[0].offset, 8192 + 16384);
    }

    #[test]
    fn depth_stencil_is_two_planes() {
        let t = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 1920,
            height: 1080,
            depth_or_array: 1,
            mips: 1,
            format: 20,
            samples: 1,
            swizzle: Swizzle::SW_64KB_Z_X,
        })
        .unwrap();
        assert_eq!(t.planes.len(), 2);
        // 4-byte depth in 128x128 blocks, 1-byte stencil in 256x256 blocks.
        assert_eq!(t.planes[0].layout.size, 1920 * 1152 * 4);
        assert_eq!(t.planes[1].offset, t.planes[0].layout.size);
        assert_eq!(t.planes[1].layout.size, 2048 * 1280);
        assert!(t.planes.iter().all(|p| p.layout.addressable()));
    }

    #[test]
    fn texture_copies_reject_overflowing_pitches() {
        let layout = Layout::new(&desc(4, 2, 1, 4, Swizzle::LINEAR)).unwrap();
        let mut tiled = vec![0; layout.size as usize];
        let mut linear = [0; 32];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(
            layout.tile(&mut tiled, 0, 0, &linear, usize::MAX, 0),
            Err(Error::TooSmall)
        );
    }

    #[test]
    fn volume_copy_rejects_overflowing_depth_pitch() {
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 3,
            ..desc(4, 2, 1, 4, Swizzle::LINEAR)
        })
        .unwrap();
        let tiled = vec![0; layout.size as usize];
        let mut linear = [0; 96];
        assert_eq!(
            layout.detile(&tiled, 0, 0, &mut linear, 16, usize::MAX),
            Err(Error::TooSmall)
        );
    }

    #[test]
    fn bad_descriptions_and_buffers_are_refused() {
        assert_eq!(
            Layout::new(&desc(0, 4, 1, 4, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 12, Swizzle::SW_4KB_S)),
            Err(Error::Invalid)
        );
        assert_eq!(
            Layout::new(&desc(4, 4, 1, 4, Swizzle(3))),
            Err(Error::Invalid)
        );
        let l = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_S)).unwrap();
        let mut out = vec![0; 64 * 64 * 4];
        assert_eq!(
            l.detile(&[0; 100], 0, 0, &mut out, 256, 0),
            Err(Error::TooSmall)
        );
        assert_eq!(l.offset(0, 64, 0, 0), Err(Error::Invalid));
    }

    /// The micro tile (256-byte) patterns written by hand from addrlib before
    /// the generated ones, by log2 element size: for each address bit the
    /// element coordinate bit it takes, `--` inside an element. Thin `S`,
    /// thin `D`, thick `S`; larger blocks add bits Y, X (thin) or Y, X, Z
    /// (thick) in turn.
    const MICRO: [[&[u8; 16]; 5]; 3] = [
        [
            b"x0x1x2x3y0y1y2y3",
            b"--x0x1x2y0y1y2x3",
            b"----x0x1y0y1y2x2",
            b"------x0y0y1x1x2",
            b"--------y0y1x0x1",
        ],
        [
            b"x0x1x2y1y0y2x3y3",
            b"--x0x1x2y0y1y2x3",
            b"----x0x1y0y1x2y2",
            b"------x0y0x1x2y1",
            b"--------x0y0x1y1",
        ],
        [
            b"x0x1z0y0z1y1x2z2",
            b"--x0z0y0z1y1x1z2",
            b"----x0y0z0y1x1z1",
            b"------x0z0y0x1z1",
            b"--------z0y0x0z1",
        ],
    ];

    /// For each axis, the address bit each coordinate bit lands on, from
    /// [`MICRO`] and the rule for the bits above it.
    fn hand_made(kind: usize, element_log2: u32, block_log2: u32) -> [Vec<u32>; 3] {
        let mut axes: [Vec<u32>; 3] = Default::default();
        let micro = MICRO[kind][element_log2 as usize];
        for bit in 0..8u32 {
            let (axis, index) = (micro[2 * bit as usize], micro[2 * bit as usize + 1]);
            let axis = match axis {
                b'x' => 0,
                b'y' => 1,
                b'z' => 2,
                _ => continue,
            };
            let index = usize::from(index - b'0');
            let to = &mut axes[axis];
            if to.len() <= index {
                to.resize(index + 1, 0);
            }
            to[index] = bit;
        }
        let order: &[usize] = if kind == 2 { &[1, 0, 2] } else { &[1, 0] };
        for (bit, axis) in (8..block_log2).zip(order.iter().cycle()) {
            axes[*axis].push(bit);
        }
        axes
    }

    /// The generated patterns of the modes without XOR are the hand-made
    /// ones, which the addresses above were checked with: a check of how the
    /// generator reads addrlib's tables.
    #[test]
    fn generated_patterns_match_the_hand_made_ones() {
        let modes = [
            (Dimension::Tex2d, Swizzle::SW_256B_S),
            (Dimension::Tex2d, Swizzle::SW_256B_D),
            (Dimension::Tex2d, Swizzle::SW_4KB_S),
            (Dimension::Tex2d, Swizzle::SW_4KB_D),
            (Dimension::Tex2d, Swizzle::SW_64KB_S),
            (Dimension::Tex2d, Swizzle::SW_64KB_D),
            (Dimension::Tex3d, Swizzle::SW_4KB_S),
            (Dimension::Tex3d, Swizzle::SW_64KB_S),
        ];
        for (dimension, swizzle) in modes {
            for e in 0..5 {
                let d = Desc {
                    dimension,
                    ..desc(1, 1, 1, 1 << e, swizzle)
                };
                for pipes in [Pipes::SERIES_X, Pipes::SERIES_S] {
                    let l = Layout::for_gpu(&d, pipes, 0x3ff).unwrap();
                    let kind = if l.thick {
                        2
                    } else {
                        usize::from(swizzle.display())
                    };
                    let hand = hand_made(kind, e, swizzle.block_log2().unwrap());
                    let flips = l.pattern.unwrap().flips;
                    for axis in 0..3 {
                        let mut want = [0; COORD_BITS];
                        for (c, &to) in hand[axis].iter().enumerate() {
                            want[c] = 1 << to;
                        }
                        assert_eq!(flips[axis], want, "{d:?} axis {axis}");
                    }
                    assert_eq!(l.xor, 0);
                }
            }
        }
    }

    /// addrlib's `ComputeOffsetFromSwizzlePattern`: each address bit is the
    /// parity of the coordinate bits its pattern entry names.
    fn pattern_offset(bits: &[u64], x: u32, y: u32, z: u32) -> u64 {
        let coords = u64::from(x) | (u64::from(y) << 16) | (u64::from(z) << 32);
        bits.iter()
            .enumerate()
            .map(|(i, &b)| u64::from((b & coords).count_ones() & 1) << i)
            .sum()
    }

    /// The generated pattern of a layout, as the generator wrote it.
    fn generated(l: &Layout) -> &'static Pattern {
        let e = l.desc.element_bytes.trailing_zeros();
        patterns::PATTERNS
            .iter()
            .find(|p| {
                p.swizzle == l.desc.swizzle
                    && p.thick == l.thick
                    && p.element_log2 == e
                    && p.pipes.is_none_or(|q| q == l.pipes)
            })
            .unwrap()
    }

    /// For every pipe configuration and element size, each pattern maps the
    /// elements of its block onto the block, each once: with the
    /// coordinates above the block at 0, and at other values (which the XOR
    /// modes mix in).
    #[test]
    fn every_pattern_is_a_permutation_of_its_block() {
        for p in patterns::PATTERNS {
            let d = Desc {
                dimension: if p.thick {
                    Dimension::Tex3d
                } else {
                    Dimension::Tex2d
                },
                ..desc(1, 1, 1, 1 << p.element_log2, p.swizzle)
            };
            let l = Layout::for_gpu(&d, p.pipes.unwrap_or(Pipes::SERIES_X), 0).unwrap();
            let [bw, bh, bd] = l.block;
            let elements = 1usize << (p.bits.len() as u32 - p.element_log2);
            assert_eq!((bw * bh * bd) as usize, elements);
            for (hx, hy, hz) in [(0, 0, 0), (5 * bw, 3 * bh, 7 * bd), (bw, 0, 2 * bd)] {
                let mut seen = vec![false; elements];
                for z in 0..bd {
                    for y in 0..bh {
                        for x in 0..bw {
                            let at = pattern_offset(p.bits, x + hx, y + hy, z + hz);
                            assert_eq!(at & ((1 << p.element_log2) - 1), 0);
                            let at = (at >> p.element_log2) as usize;
                            assert!(!seen[at], "{d:?} {:?} ({x},{y},{z}) twice", p.pipes);
                            seen[at] = true;
                        }
                    }
                }
            }
        }
    }

    /// `Layout::offset` gives the addresses of addrlib's
    /// `ComputeSurfaceAddrFromCoordMacroTiled`, written out here from the
    /// generated patterns alone: the slice (or thick slab), the mip, the
    /// block, then the pattern of the coordinates (shifted into the tail
    /// for tail mips) XOR the base swizzle's pipe and bank bits.
    #[test]
    fn xor_addresses_follow_addrlib() {
        let cases = [
            desc(300, 200, 9, 4, Swizzle::SW_64KB_R_X),
            desc(513, 40, 10, 1, Swizzle::SW_64KB_R_X),
            desc(200, 300, 9, 16, Swizzle::SW_64KB_R_X),
            desc(260, 66, 9, 2, Swizzle::SW_64KB_Z_X),
            desc(150, 90, 8, 8, Swizzle::SW_64KB_S_X),
            desc(150, 90, 8, 2, Swizzle::SW_64KB_D_X),
            desc(90, 50, 7, 1, Swizzle::SW_64KB_S_T),
            desc(77, 99, 7, 16, Swizzle::SW_64KB_D_T),
            desc(61, 45, 6, 4, Swizzle::SW_4KB_S_X),
            desc(50, 70, 7, 8, Swizzle::SW_4KB_D_X),
            Desc {
                depth: 9,
                ..desc(70, 70, 2, 4, Swizzle::SW_64KB_D_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 40,
                ..desc(40, 36, 6, 2, Swizzle::SW_64KB_S_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 9,
                ..desc(33, 20, 5, 16, Swizzle::SW_64KB_D_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 12,
                ..desc(30, 30, 5, 4, Swizzle::SW_4KB_S_X)
            },
            Desc {
                dimension: Dimension::Tex3d,
                depth: 6,
                ..desc(70, 40, 4, 1, Swizzle::SW_64KB_R_X)
            },
        ];
        for d in cases {
            for (pipes, base) in [(Pipes::SERIES_X, 0x5a5), (Pipes::SERIES_S, 0x0d3)] {
                let l = Layout::for_gpu(&d, pipes, base).unwrap();
                let bits = generated(&l).bits;
                let log2 = l.alignment.trailing_zeros();
                let [bw, bh, bd] = l.block;
                // GetBankXorBits: banks above the pipes and two column bits.
                let p = pipes.pipes_log2;
                let banks = if log2 > 10 + p {
                    (log2 - 10 - p).min(4)
                } else {
                    0
                };
                let mask = ((1 << p) - 1) | (((1 << banks) - 1) << (p + 2));
                let xor = (u64::from(base & mask) << 8) & ((1 << log2) - 1);
                for (mip, m) in l.mips.iter().enumerate() {
                    let depth = if d.dimension == Dimension::Tex3d {
                        m.depth
                    } else {
                        d.depth
                    };
                    let (tx, ty) = m.tail.map_or((0, 0), |t| (t.x, t.y));
                    for z in 0..depth {
                        let slice = if l.thick { z / bd } else { z };
                        for y in 0..m.height {
                            for x in 0..m.width {
                                let block = u64::from(y / bh * (m.pitch / bw) + x / bw);
                                let want = u64::from(slice) * l.slice_size
                                    + m.offset
                                    + (block << log2)
                                    + (pattern_offset(bits, x + tx, y + ty, z) ^ xor);
                                let got = l.offset(mip as u32, x, y, z);
                                assert_eq!(
                                    got,
                                    Ok(want),
                                    "{d:?} {pipes:?} mip {mip} ({x},{y},{z})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Hand-checked from addrlib's `GFX10_SW_64K_R_X_1xaa_RBPLUS_PATINFO`
    /// row for 32 pipes, 8 packers, 4-byte elements (nibbles 39, 307, 674):
    /// address bits 8-15 are y4^x7^y7, z2^x4^y4, z1^y5^x6, z0^x5^y6,
    /// x6^x8^y8, x3, y3, y6^x7. Blocks are 128 x 128.
    #[test]
    fn hand_checked_xor_addresses() {
        let l = Layout::new(&Desc {
            depth: 2,
            ..desc(512, 256, 1, 4, Swizzle::SW_64KB_R_X)
        })
        .unwrap();
        assert_eq!(l.block, [128, 128, 1]);
        assert_eq!(l.offset(0, 1, 0, 0), Ok(1 << 2));
        assert_eq!(l.offset(0, 0, 16, 0), Ok((1 << 8) | (1 << 9)));
        // x7 is the next block, and flips bits 8 and 15 inside it.
        assert_eq!(l.offset(0, 128, 0, 0), Ok(65536 | (1 << 8) | (1 << 15)));
        // y = 128: the row of blocks below (4 blocks a row); y7 flips bit 8.
        assert_eq!(l.offset(0, 0, 128, 0), Ok((4 * 65536) | (1 << 8)));
        // The array slice flips bit 11 (z0).
        assert_eq!(l.offset(0, 0, 0, 1), Ok(l.slice_size | (1 << 11)));
        // A base swizzle of pipe 1 flips bit 8 of every element.
        let l2 = Layout::for_gpu(&l.desc, Pipes::SERIES_X, 1).unwrap();
        assert_eq!(l2.offset(0, 0, 16, 0), Ok(1 << 9));
        // 64 KiB blocks of 32 pipes have one bank bit: base swizzle bit 7
        // lands on address bit 15; bits above it are ignored.
        let l3 = Layout::for_gpu(&l.desc, Pipes::SERIES_X, 0x180).unwrap();
        assert_eq!(l3.offset(0, 0, 0, 0), Ok(1 << 15));
    }

    /// What lets copies repeat one row's runs on every row: the lowest
    /// address bit a row can flip (its y or slice bits, or the base swizzle
    /// from bit 8 up) is one no column flips. Columns never set it, so a run
    /// of consecutive column offsets cannot cross it, and a row's XOR moves
    /// each run whole.
    #[test]
    fn rows_flip_a_low_bit_of_their_own() {
        for p in patterns::PATTERNS {
            let d = Desc {
                dimension: if p.thick {
                    Dimension::Tex3d
                } else {
                    Dimension::Tex2d
                },
                ..desc(1, 1, 1, 1 << p.element_log2, p.swizzle)
            };
            let l = Layout::for_gpu(&d, p.pipes.unwrap_or(Pipes::SERIES_X), u32::MAX).unwrap();
            let flips = l.pattern.unwrap().flips;
            let rows = flips[1..].as_flattened().iter().fold(l.xor, |a, &f| a | f);
            let columns = flips[0].iter().fold(0, |a, &f| a | f);
            let lowest = rows & rows.wrapping_neg();
            assert_eq!(columns & lowest, 0, "{d:?} {:?}", p.pipes);
        }
    }

    /// The pipe configuration changes the XOR modes' addresses, not their
    /// sizes.
    #[test]
    fn series_x_and_series_s_differ_for_64kb_r_x() {
        let d = desc(256, 256, 1, 4, Swizzle::SW_64KB_R_X);
        let x = Layout::for_gpu(&d, Pipes::SERIES_X, 0).unwrap();
        let s = Layout::for_gpu(&d, Pipes::SERIES_S, 0).unwrap();
        assert_eq!((x.size, x.block), (s.size, s.block));
        let mut differ = 0;
        for y in 0..256 {
            for c in 0..256 {
                differ += usize::from(x.offset(0, c, y, 0) != s.offset(0, c, y, 0));
            }
        }
        assert!(differ > 256 * 256 / 2, "{differ}");
        // Without XOR the configuration plays no part.
        let d = desc(256, 256, 1, 4, Swizzle::SW_64KB_D);
        let x = Layout::for_gpu(&d, Pipes::SERIES_X, 7).unwrap();
        let s = Layout::for_gpu(&d, Pipes::SERIES_S, 7).unwrap();
        assert_eq!((x.pattern, x.xor), (s.pattern, 0));
        assert!(std::ptr::eq(x.pattern.unwrap(), s.pattern.unwrap()));
        assert!(std::ptr::eq(x.pattern.unwrap(), x.clone().pattern.unwrap()));
        // addrlib has no 4KB_R_X pattern: such layouts have a size only.
        let l = Layout::new(&desc(64, 64, 1, 4, Swizzle::SW_4KB_R_X)).unwrap();
        assert!(!l.addressable());
        assert_eq!(l.offset(0, 0, 0, 0), Err(Error::Unsupported));
    }

    #[test]
    fn shared_run_offsets_cover_shifted_mips_and_rows_spanning_tiles() {
        let mut bc7 = desc(521, 317, 9, 16, Swizzle::SW_64KB_D);
        bc7.block = 4;
        let mut vol = desc(35, 19, 5, 2, Swizzle::SW_4KB_S_X);
        vol.dimension = Dimension::Tex3d;
        vol.depth = 11;
        let cases = [
            (2, desc(301, 173, 8, 1, Swizzle::SW_64KB_Z_X)),
            (4, vol),
            (8, desc(261, 131, 8, 1, Swizzle::SW_64KB_D)),
            (16, desc(515, 259, 9, 4, Swizzle::SW_64KB_R_X)),
            (32, bc7),
        ];
        for (want_run, d) in cases {
            let layout = Layout::for_gpu(&d, Pipes::SERIES_X, 0x1f3).unwrap();
            let flips = &layout.pattern.unwrap().flips;
            let (mut borrowed_shifted, mut multiple_tiles, mut partial_run) = (false, false, false);
            for mip in &layout.mips {
                let (bytes, offsets) = layout.row_runs(mip).unwrap();
                assert_eq!(bytes, want_run, "{d:?}");
                borrowed_shifted |=
                    matches!(offsets, Cow::Borrowed(_)) && mip.tail.is_some_and(|t| t.x != 0);
                multiple_tiles |= matches!(offsets, Cow::Owned(_));
                let elements = bytes as u32 / layout.desc.element_bytes;
                partial_run |= !mip.width.is_multiple_of(elements);
                for x in 0..mip.width {
                    let within = (x % elements) * layout.desc.element_bytes;
                    assert_eq!(
                        offsets[(x / elements) as usize] as u64 + u64::from(within),
                        layout.column(flips, mip, x),
                    );
                }
            }
            assert!(borrowed_shifted && multiple_tiles && partial_run, "{d:?}");
            let bpe = d.element_bytes as usize;
            let m = &layout.mips[0];
            let active = m.width as usize * bpe;
            let (row, plane) = (active + 7, (active + 7) * m.height as usize + 5);
            let linear: Vec<u8> = (0..plane * m.depth as usize)
                .map(|i| (i.wrapping_mul(29) ^ (want_run * 7)) as u8)
                .collect();
            let mut tiled = vec![0x5au8; layout.size as usize];
            layout.tile(&mut tiled, 0, 0, &linear, row, plane).unwrap();
            let mut back = vec![0xa5u8; linear.len()];
            layout.detile(&tiled, 0, 0, &mut back, row, plane).unwrap();
            for z in 0..m.depth {
                for y in 0..m.height {
                    let start = z as usize * plane + y as usize * row;
                    assert_eq!(back[start..start + active], linear[start..start + active]);
                    assert!(back[start + active..start + row].iter().all(|&b| b == 0xa5));
                }
            }
        }
    }

    /// Detile speed per mode on one core:
    /// `cargo test --release -p xtp-formats detile_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn detile_throughput() {
        let modes = [
            Swizzle::SW_4KB_S,
            Swizzle::SW_64KB_D,
            Swizzle::SW_4KB_S_X,
            Swizzle::SW_4KB_D_X,
            Swizzle::SW_64KB_S_T,
            Swizzle::SW_64KB_D_T,
            Swizzle::SW_64KB_Z_X,
            Swizzle::SW_64KB_S_X,
            Swizzle::SW_64KB_D_X,
            Swizzle::SW_64KB_R_X,
        ];
        for swizzle in modes {
            for bytes in [4, 16] {
                let l = Layout::new(&desc(2048, 2048, 1, bytes, swizzle)).unwrap();
                let tiled: Vec<u8> = (0..l.size).map(|i| i as u8).collect();
                let row = 2048 * bytes as usize;
                let mut linear = vec![0u8; row * 2048];
                let runs = 10;
                let started = std::time::Instant::now();
                for _ in 0..runs {
                    l.detile(std::hint::black_box(&tiled), 0, 0, &mut linear, row, 0)
                        .unwrap();
                }
                let secs = started.elapsed().as_secs_f64() / f64::from(runs);
                println!(
                    "{swizzle:?} {bytes} B: {:.0} MB/s",
                    linear.len() as f64 / secs / 1e6
                );
            }
        }
    }
}
