//! Byte-scan SIMD primitives.
//!
//! Two functions, three implementations each (NEON on aarch64, AVX2 on
//! x86_64, scalar everywhere). The public API is safe; runtime CPU
//! feature detection chooses the impl per call on x86_64. NEON is part of
//! the aarch64 base ISA so no detection is needed there.

/// Find the first occurrence of `needle` in `haystack`.
pub fn scan_for_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::scan_for_byte(haystack, needle) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { avx2::scan_for_byte(haystack, needle) };
        }
    }
    #[allow(unreachable_code)]
    scalar::scan_for_byte(haystack, needle)
}

/// Find the first position `i` where `haystack[i] == a` and `haystack[i+1] == b`.
pub fn scan_for_pair(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::scan_for_pair(haystack, a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { avx2::scan_for_pair(haystack, a, b) };
        }
    }
    #[allow(unreachable_code)]
    scalar::scan_for_pair(haystack, a, b)
}

pub(crate) mod scalar {
    pub fn scan_for_byte(haystack: &[u8], needle: u8) -> Option<usize> {
        haystack.iter().position(|&b| b == needle)
    }

    pub fn scan_for_pair(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
        haystack.windows(2).position(|w| w[0] == a && w[1] == b)
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_for_byte(haystack: &[u8], needle: u8) -> Option<usize> {
        let v_needle = _mm256_set1_epi8(needle as i8);
        let mut i = 0;
        while i + 32 <= haystack.len() {
            let v = unsafe { _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i) };
            let cmp = _mm256_cmpeq_epi8(v, v_needle);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
        while i < haystack.len() {
            if haystack[i] == needle {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_for_pair(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
        if haystack.len() < 2 {
            return None;
        }
        let va = _mm256_set1_epi8(a as i8);
        let vb = _mm256_set1_epi8(b as i8);
        let mut i = 0;
        // Load 32 bytes at i for `a` and 32 bytes at i+1 for `b`; needs i+33 <= len.
        while i + 33 <= haystack.len() {
            let v0 = unsafe { _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i) };
            let v1 =
                unsafe { _mm256_loadu_si256(haystack.as_ptr().add(i + 1) as *const __m256i) };
            let m0 = _mm256_cmpeq_epi8(v0, va);
            let m1 = _mm256_cmpeq_epi8(v1, vb);
            let m = _mm256_and_si256(m0, m1);
            let mask = _mm256_movemask_epi8(m) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
        while i + 1 < haystack.len() {
            if haystack[i] == a && haystack[i + 1] == b {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Reduce a 128-bit byte-comparison vector to a 64-bit mask where each
    /// pair of input bytes collapses to one output nibble. Canonical NEON
    /// "movemask" replacement; `vshrn_n_u16(v, 4)` is one cycle on modern
    /// Arm cores.
    #[target_feature(enable = "neon")]
    unsafe fn first_match_pos(cmp: uint8x16_t) -> Option<usize> {
        let narrow = vshrn_n_u16::<4>(vreinterpretq_u16_u8(cmp));
        let mask: u64 = vget_lane_u64::<0>(vreinterpret_u64_u8(narrow));
        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize / 4)
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn scan_for_byte(haystack: &[u8], needle: u8) -> Option<usize> {
        let v_needle = vdupq_n_u8(needle);
        let mut i = 0;
        while i + 16 <= haystack.len() {
            let v = unsafe { vld1q_u8(haystack.as_ptr().add(i)) };
            let cmp = vceqq_u8(v, v_needle);
            if let Some(off) = unsafe { first_match_pos(cmp) } {
                return Some(i + off);
            }
            i += 16;
        }
        while i < haystack.len() {
            if haystack[i] == needle {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn scan_for_pair(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
        if haystack.len() < 2 {
            return None;
        }
        let va = vdupq_n_u8(a);
        let vb = vdupq_n_u8(b);
        let mut i = 0;
        while i + 17 <= haystack.len() {
            let v0 = unsafe { vld1q_u8(haystack.as_ptr().add(i)) };
            let v1 = unsafe { vld1q_u8(haystack.as_ptr().add(i + 1)) };
            let m0 = vceqq_u8(v0, va);
            let m1 = vceqq_u8(v1, vb);
            let m = vandq_u8(m0, m1);
            if let Some(off) = unsafe { first_match_pos(m) } {
                return Some(i + off);
            }
            i += 16;
        }
        while i + 1 < haystack.len() {
            if haystack[i] == a && haystack[i + 1] == b {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross_check_byte(haystack: &[u8], needle: u8) {
        let got = scan_for_byte(haystack, needle);
        let want = scalar::scan_for_byte(haystack, needle);
        assert_eq!(got, want, "len={} needle={:#x}", haystack.len(), needle);
    }

    fn cross_check_pair(haystack: &[u8], a: u8, b: u8) {
        let got = scan_for_pair(haystack, a, b);
        let want = scalar::scan_for_pair(haystack, a, b);
        assert_eq!(got, want, "len={} a={:#x} b={:#x}", haystack.len(), a, b);
    }

    #[test]
    fn byte_empty_and_single() {
        cross_check_byte(&[], b'x');
        cross_check_byte(b"x", b'x');
        cross_check_byte(b"y", b'x');
    }

    #[test]
    fn byte_every_position_and_length() {
        for len in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255] {
            let mut buf = vec![b'.'; len];
            for pos in 0..len {
                buf[pos] = b'x';
                assert_eq!(scan_for_byte(&buf, b'x'), Some(pos), "len={len} pos={pos}");
                buf[pos] = b'.';
            }
            assert_eq!(scan_for_byte(&buf, b'x'), None, "absent len={len}");
        }
    }

    #[test]
    fn byte_random_distribution() {
        let mut buf = Vec::with_capacity(8192);
        let mut state: u32 = 0x9e3779b9;
        for _ in 0..8192 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            buf.push((state >> 24) as u8);
        }
        for &needle in &[0u8, 1, 0x7f, 0x80, 0xff] {
            cross_check_byte(&buf, needle);
        }
    }

    #[test]
    fn pair_basic_and_sse_framing() {
        cross_check_pair(b"hello {{world}}", b'{', b'{');
        cross_check_pair(b"no pair here", b'{', b'{');
        cross_check_pair(b"{", b'{', b'{');
        cross_check_pair(b"{{", b'{', b'{');
        cross_check_pair(b"data: a\n\ndata: b\n\n", b'\n', b'\n');
    }

    #[test]
    fn pair_every_position_and_length() {
        for len in [2usize, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128] {
            for pos in 0..len - 1 {
                let mut buf = vec![b'.'; len];
                buf[pos] = b'<';
                buf[pos + 1] = b'/';
                assert_eq!(
                    scan_for_pair(&buf, b'<', b'/'),
                    Some(pos),
                    "len={len} pos={pos}"
                );
            }
        }
    }

    #[test]
    fn pair_overlap_does_not_double_count() {
        assert_eq!(scan_for_pair(b"{{{", b'{', b'{'), Some(0));
    }

    #[test]
    fn pair_same_byte_repeated() {
        let buf = vec![b'\n'; 100];
        assert_eq!(scan_for_pair(&buf, b'\n', b'\n'), Some(0));
    }
}
