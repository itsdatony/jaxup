// The MIT License (MIT)
//
// Copyright (c) 2017-2018 Kyle Hawk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef JAXUP_GRISU_H
#define JAXUP_GRISU_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace jaxup {
namespace grisu {

class ExplodedFloatingPoint {
public:
	uint64_t mantissa;
	int exponent;

	ExplodedFloatingPoint() = default;
	ExplodedFloatingPoint(const double d) {
		assert(d > 0.0);
		uint64_t bitString = doubleAsU64(d);
		uint64_t significand = bitString & significandMask;
		int biasedExponent = bitString >> 52; // no need to mask sign bit b/c it's 0
		if (biasedExponent != 0) {
			mantissa = significand + impliedBitOffset;
			exponent = biasedExponent - exponentBias;
		} else {
			mantissa = significand;
			exponent = 1 - exponentBias;
		}
	}
	ExplodedFloatingPoint(const uint64_t mantissa, const int exponent) : mantissa{mantissa}, exponent{exponent} {}
	ExplodedFloatingPoint operator-(const ExplodedFloatingPoint& rhs) const {
		assert(rhs.exponent == exponent);
		return {mantissa - rhs.mantissa, exponent};
	}

	ExplodedFloatingPoint operator*(const ExplodedFloatingPoint& rhs) const {
		static constexpr uint64_t roundUp32 = 1ULL << 31;
		uint64_t topL = top32(mantissa);
		uint64_t botL = bot32(mantissa);
		uint64_t topR = top32(rhs.mantissa);
		uint64_t botR = bot32(rhs.mantissa);
		uint64_t tops = topL * topR;
		uint64_t mid1 = botL * topR;
		uint64_t mid2 = topL * botR;
		uint64_t bots = botL * botR;
		uint64_t tmp = top32(bots) + bot32(mid1) + bot32(mid2) + roundUp32;
		return {tops + top32(mid1) + top32(mid2) + top32(tmp), exponent + rhs.exponent + 64};
	}

	void normalize(const unsigned int offset = 0) {
		uint64_t neededBit = impliedBitOffset << offset;
		while ((mantissa & neededBit) == 0 && mantissa > 0) {
			mantissa <<= 1;
			--exponent;
		}

		const int extraBits = explodedSignificandSizeBits - (significandSizeBits + 1 + offset);
		mantissa <<= extraBits;
		exponent -= extraBits;
	}

	void getBounds(ExplodedFloatingPoint& minus, ExplodedFloatingPoint& plus) {
		plus.mantissa = (mantissa << 1) + 1;
		plus.exponent = exponent - 1;
		plus.normalize(1);
		int minusOffset = (mantissa == impliedBitOffset) ? 2 : 1;
		minus.mantissa = (mantissa << minusOffset) - 1;
		minus.exponent = exponent - minusOffset;
		// Copy plus's normalization
		minus.mantissa <<= minus.exponent - plus.exponent;
		minus.exponent = plus.exponent;
	}

private:
	static constexpr int significandSizeBits = 52;
	static constexpr int explodedSignificandSizeBits = 64;
	static constexpr int exponentBias = 1075;
	static constexpr uint64_t significandMask = 0x000FFFFFFFFFFFFF;
	static constexpr uint64_t impliedBitOffset = 1ULL << significandSizeBits;

	typedef union {
		double dub;
		uint64_t u64;
	} DoubleConverter;

	static_assert(sizeof(double) == sizeof(uint64_t), "Double precision floating point values are expected to be 64-bits wide");
	inline uint64_t doubleAsU64(const double d) const {
		DoubleConverter converter;
		converter.dub = d;
		return converter.u64;
	}

	inline double u64AsDouble(const uint64_t u64) const {
		DoubleConverter converter;
		converter.u64 = u64;
		return converter.dub;
	}

	inline constexpr uint64_t top32(uint64_t val) const {
		return val >> 32;
	}

	inline constexpr uint64_t bot32(uint64_t val) const {
		return val & 0xFFFFFFFF;
	}
};

static const ExplodedFloatingPoint powerCache[] = {
	{0xfa8fd5a0081c0288, -1220},
	{0xbaaee17fa23ebf76, -1193},
	{0x8b16fb203055ac76, -1166},
	{0xcf42894a5dce35ea, -1140},
	{0x9a6bb0aa55653b2d, -1113},
	{0xe61acf033d1a45df, -1087},
	{0xab70fe17c79ac6ca, -1060},
	{0xff77b1fcbebcdc4f, -1034},
	{0xbe5691ef416bd60c, -1007},
	{0x8dd01fad907ffc3c, -980},
	{0xd3515c2831559a83, -954},
	{0x9d71ac8fada6c9b5, -927},
	{0xea9c227723ee8bcb, -901},
	{0xaecc49914078536d, -874},
	{0x823c12795db6ce57, -847},
	{0xc21094364dfb5637, -821},
	{0x9096ea6f3848984f, -794},
	{0xd77485cb25823ac7, -768},
	{0xa086cfcd97bf97f4, -741},
	{0xef340a98172aace5, -715},
	{0xb23867fb2a35b28e, -688},
	{0x84c8d4dfd2c63f3b, -661},
	{0xc5dd44271ad3cdba, -635},
	{0x936b9fcebb25c996, -608},
	{0xdbac6c247d62a584, -582},
	{0xa3ab66580d5fdaf6, -555},
	{0xf3e2f893dec3f126, -529},
	{0xb5b5ada8aaff80b8, -502},
	{0x87625f056c7c4a8b, -475},
	{0xc9bcff6034c13053, -449},
	{0x964e858c91ba2655, -422},
	{0xdff9772470297ebd, -396},
	{0xa6dfbd9fb8e5b88f, -369},
	{0xf8a95fcf88747d94, -343},
	{0xb94470938fa89bcf, -316},
	{0x8a08f0f8bf0f156b, -289},
	{0xcdb02555653131b6, -263},
	{0x993fe2c6d07b7fac, -236},
	{0xe45c10c42a2b3b06, -210},
	{0xaa242499697392d3, -183},
	{0xfd87b5f28300ca0e, -157},
	{0xbce5086492111aeb, -130},
	{0x8cbccc096f5088cc, -103},
	{0xd1b71758e219652c, -77},
	{0x9c40000000000000, -50},
	{0xe8d4a51000000000, -24},
	{0xad78ebc5ac620000, 3},
	{0x813f3978f8940984, 30},
	{0xc097ce7bc90715b3, 56},
	{0x8f7e32ce7bea5c70, 83},
	{0xd5d238a4abe98068, 109},
	{0x9f4f2726179a2245, 136},
	{0xed63a231d4c4fb27, 162},
	{0xb0de65388cc8ada8, 189},
	{0x83c7088e1aab65db, 216},
	{0xc45d1df942711d9a, 242},
	{0x924d692ca61be758, 269},
	{0xda01ee641a708dea, 295},
	{0xa26da3999aef774a, 322},
	{0xf209787bb47d6b85, 348},
	{0xb454e4a179dd1877, 375},
	{0x865b86925b9bc5c2, 402},
	{0xc83553c5c8965d3d, 428},
	{0x952ab45cfa97a0b3, 455},
	{0xde469fbd99a05fe3, 481},
	{0xa59bc234db398c25, 508},
	{0xf6c69a72a3989f5c, 534},
	{0xb7dcbf5354e9bece, 561},
	{0x88fcf317f22241e2, 588},
	{0xcc20ce9bd35c78a5, 614},
	{0x98165af37b2153df, 641},
	{0xe2a0b5dc971f303a, 667},
	{0xa8d9d1535ce3b396, 694},
	{0xfb9b7cd9a4a7443c, 720},
	{0xbb764c4ca7a44410, 747},
	{0x8bab8eefb6409c1a, 774},
	{0xd01fef10a657842c, 800},
	{0x9b10a4e5e9913129, 827},
	{0xe7109bfba19c0c9d, 853},
	{0xac2820d9623bf429, 880},
	{0x80444b5e7aa7cf85, 907},
	{0xbf21e44003acdd2d, 933},
	{0x8e679c2f5e44ff8f, 960},
	{0xd433179d9c8cb841, 986},
	{0x9e19db92b4e31ba9, 1013},
	{0xeb96bf6ebadf77d9, 1039},
	{0xaf87023b9bf0ee6b, 1066}};

inline const ExplodedFloatingPoint& getCachedPower(int min, int& powTen) {
	static const double baseChange = 1.0 / std::log2(10.0);
	int k = static_cast<int>(std::ceil((min + 63) * baseChange)) + 347;
	int i = (k >> 3) + 1;
	assert(i >= 0 && static_cast<unsigned int>(i) < (sizeof(powerCache) / sizeof(powerCache[0])));
	powTen = 348 - (i << 3);
	return powerCache[i];
}

inline uint64_t getIntegerPowTen(int exponent) {
	assert(exponent >= 0 && exponent < 20);
	static const uint64_t powers[] = {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL, 10000000000000ULL, 100000000000000ULL, 1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL, 1000000000000000000ULL, 10000000000000000000ULL};
	return powers[exponent];
}

inline double raiseToPowTen(double base, int powTen) {
	static constexpr double powers[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39, 1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47, 1e48, 1e49, 1e50, 1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59, 1e60, 1e61, 1e62, 1e63, 1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 1e70, 1e71, 1e72, 1e73, 1e74, 1e75, 1e76, 1e77, 1e78, 1e79, 1e80, 1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87, 1e88, 1e89, 1e90, 1e91, 1e92, 1e93, 1e94, 1e95, 1e96, 1e97, 1e98, 1e99, 1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109, 1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119, 1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129, 1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159, 1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169, 1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189, 1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199, 1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209, 1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219, 1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239, 1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249, 1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279, 1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289, 1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299, 1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308};
	if (powTen < 0) {
		if (powTen < -308) {
			if (powTen < -(308 * 2)) {
				return 0.0;
			}
			base *= 1e-308;
			powTen += 308;
		}
		return base / powers[-powTen];
	}
	if (powTen > 308) {
		powTen = 308;
	}
	return base * powers[powTen];
}

inline constexpr char digitToAscii(char d) {
	return '0' + d;
}

inline void generateDigits(ExplodedFloatingPoint& plus, uint64_t delta, char* buffer, int& len, int& powTen) {
	ExplodedFloatingPoint one(1ULL << -plus.exponent, plus.exponent);
	uint32_t part1 = static_cast<uint32_t>(plus.mantissa >> -one.exponent);
	uint64_t part2 = plus.mantissa & (one.mantissa - 1);
	len = 0;
	int kappa = 10;
	uint32_t div = 1000000000, d;
	while (kappa > 0) {
		d = part1 / div;
		if (d != 0 || len != 0) {
			buffer[len++] = digitToAscii(d);
		}
		part1 %= div;
		--kappa;
		div /= 10;
		if ((((uint64_t)part1) << -one.exponent) + part2 <= delta) {
			powTen += kappa;
			return;
		}
	}
	do {
		part2 *= 10;
		d = static_cast<uint32_t>(part2 >> -one.exponent);
		if (d != 0 || len != 0) {
			buffer[len++] = digitToAscii(d);
		}
		part2 &= (one.mantissa - 1);
		--kappa;
		delta *= 10;
	} while (part2 > delta);
	powTen += kappa;
}

inline void grisu2(const double d, char* buffer, int& length, int& powTen) {
	ExplodedFloatingPoint plus, minus, v{d};
	v.getBounds(minus, plus);
	const auto& cached = getCachedPower(-59 - (plus.exponent + 64), powTen);
	auto wPlus = cached * plus;
	auto wMinus = cached * minus;
	++wPlus.mantissa;
	--wMinus.mantissa;
	generateDigits(wPlus, wPlus.mantissa - wMinus.mantissa, buffer, length, powTen);
}

inline int writeExponent(char* buffer, int powTen) {
	if (powTen < 0) {
		buffer[0] = '-';
		return 1 + writeExponent(buffer + 1, -powTen);
	}
	if (powTen >= 100) {
		buffer[0] = digitToAscii(powTen / 100);
		powTen %= 100;
		buffer[1] = digitToAscii(powTen / 10);
		buffer[2] = digitToAscii(powTen % 10);
		return 3;
	} else if (powTen >= 10) {
		buffer[0] = digitToAscii(powTen / 10);
		buffer[1] = digitToAscii(powTen % 10);
		return 2;
	} else {
		buffer[0] = digitToAscii(powTen);
		return 1;
	}
}

inline int conformalizeNumberString(char* buffer, int length, int powTen) {
	const int totalPowTen = length + powTen;
	if (totalPowTen <= 19) {
		if (powTen >= 0) {
			// Whole number with no exponent
			// add trailing zeros
			for (int i = length; i < totalPowTen; ++i) {
				buffer[i] = '0';
			}
			return totalPowTen;
		} else if (totalPowTen > 0) {
			// Decimal number with no exponent
			// make room for decimal point and then insert it
			std::memmove(&buffer[totalPowTen + 1], &buffer[totalPowTen], length - totalPowTen);
			buffer[totalPowTen] = '.';
			return length + 1;
		} else if (totalPowTen > -6) {
			// Short decimal < 1
			// Make room for '0.' + any preceding zeros
			const int offset = 2 - totalPowTen;
			std::memmove(&buffer[offset], buffer, length);
			buffer[0] = '0';
			buffer[1] = '.';
			for (int i = 2; i < offset; ++i) {
				buffer[i] = '0';
			}
			return offset + length;
		}
	}
	// Use scientific notation
	if (length == 1) {
		buffer[1] = 'e';
		return 2 + writeExponent(&buffer[2], powTen);
	} else {
		// make room for conventional decimal and then insert it
		std::memmove(&buffer[2], &buffer[1], length - 1);
		buffer[1] = '.';
		buffer[length + 1] = 'e';
		return length + 2 + writeExponent(&buffer[length + 2], totalPowTen - 1);
	}
}

inline int fastDoubleToString(char* buffer, const double d) {
	if (d == 0.0) {
		buffer[0] = '0';
		return 1;
	}
	if (std::signbit(d)) {
		buffer[0] = '-';
		return 1 + fastDoubleToString(buffer + 1, std::fabs(d));
	}
	int length;
	int powTen;
	grisu2(d, buffer, length, powTen);
	return conformalizeNumberString(buffer, length, powTen);
}
}
}

#endif
