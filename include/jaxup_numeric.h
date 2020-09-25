// The MIT License (MIT)
//
// Copyright (c) 2017-2020 Kyle Hawk
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

#ifndef JAXUP_NUMERIC_H
#define JAXUP_NUMERIC_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "jaxup_power_tables.h"

namespace jaxup {
namespace numeric {

inline char* writeUnsignedIntegerToBuff(uint64_t value, char* endMarker) {
	static const char digits[] = "00102030405060708090011121314151617181910212223242526272829203132333435363738393041424344454647484940515253545556575859506162636465666768696071727374757677787970818283848586878889809192939495969798999";
	unsigned int offset;
	char* start = endMarker;
	while (value >= 100) {
		offset = (value % 100) * 2;
		value = value / 100;
		*--start = digits[offset];
		*--start = digits[offset + 1];
	}
	if (value < 10) {
		*--start = '0' + static_cast<char>(value);
		return start;
	}
	offset = static_cast<unsigned int>(value) * 2;
	*--start = digits[offset];
	*--start = digits[offset + 1];
	return start;
}

inline char* writeIntegerToBuff(int64_t value, char* endMarker) {
	if (value >= 0) {
		return writeUnsignedIntegerToBuff((uint64_t)value, endMarker);
	} else {
		char* start = writeUnsignedIntegerToBuff((uint64_t)(0 - value), endMarker);
		*--start = '-';
		return start;
	}
}

class ExplodedFloatingPoint {
public:
	uint64_t mantissa;
	int exponent;

	ExplodedFloatingPoint() = default;
	ExplodedFloatingPoint(const ExplodedFloatingPoint&) = default;
	ExplodedFloatingPoint(const double d, bool normalize = false) {
		assert(d > 0.0);
		uint64_t bitString = doubleAsU64(d);
		uint64_t significand = bitString & significandMask;
		int biasedExponent = bitString >> significandSizeBits; // no need to mask sign bit b/c it's 0
		if (biasedExponent != 0) {
			mantissa = significand + impliedBitOffset;
			exponent = biasedExponent - exponentBias;
		} else {
			mantissa = significand;
			exponent = 1 - exponentBias;
		}
		if (normalize) {
			this->normalize();
		}
	}
	ExplodedFloatingPoint(const uint64_t mantissa, const int exponent) : mantissa{mantissa}, exponent{exponent} {}

	double asDouble() const {
		auto temp = ExplodedFloatingPoint(mantissa, exponent);
		if (temp.mantissa == 0) {
			return 0.0;
		}
		temp.normalize(11);

		int64_t biasedExponent = temp.exponent + exponentBias + 11;
		uint64_t out;
		assert(biasedExponent < 1024 && biasedExponent > -52);
		if (biasedExponent > 0) {
			out = temp.mantissa >> 10;
			// round to even
			out += (out & 1) && ((out & 2) || (temp.mantissa & 0x3FF));
			if (out & (impliedBitOffset << 2)) {
				out = out >> 1;
				++biasedExponent;
			}
			if (biasedExponent >= 2047) {
				return std::numeric_limits<double>::infinity();
			}
			out = ((out >> 1) & significandMask) | (biasedExponent << significandSizeBits);
		} else {
			// denormal case
			if (biasedExponent <= -53) {
				return 0.0;
			}
			out = temp.mantissa >> (11 - biasedExponent);
			// round to even
			out += (out & 1) && ((out & 2) || (temp.mantissa & (0xFFFFFFFFFFFFFFFFULL >> (53 + biasedExponent))));
			// overflow happily rolls into a normalized number, so no need to check
			out = out >> 1;
		}
		return u64AsDouble(out);
	}

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

	int normalize(const unsigned int offset = 0) {
		uint64_t neededBit = impliedBitOffset << offset;
		while ((mantissa & neededBit) == 0 && mantissa > 0) {
			mantissa <<= 1;
			--exponent;
		}

		const int extraBits = explodedSignificandSizeBits - (significandSizeBits + 1 + offset);
		if (extraBits != 0) {
			mantissa <<= extraBits;
			exponent -= extraBits;
		}
		return extraBits;
	}

private:
	static constexpr int significandSizeBits = 52;
	static constexpr int explodedSignificandSizeBits = 64;
	static constexpr int exponentBias = 1075;
	static constexpr uint64_t significandMask = 0x000FFFFFFFFFFFFF;
	static constexpr uint64_t impliedBitOffset = 1ULL << significandSizeBits;

	static_assert(sizeof(double) == sizeof(uint64_t), "Double precision floating point values are expected to be 64-bits wide");
	static inline uint64_t doubleAsU64(const double d) {
		uint64_t u64;
		std::memcpy(&u64, &d, sizeof(d));
		return u64;
	}

	static inline double u64AsDouble(const uint64_t u64) {
		double d;
		std::memcpy(&d, &u64, sizeof(u64));
		return d;
	}

	static inline constexpr uint64_t top32(uint64_t val) {
		return val >> 32;
	}

	static inline constexpr uint64_t bot32(uint64_t val) {
		return val & 0xFFFFFFFF;
	}
};

// Cache of every 8th power of 10 (i.e. 1e-348, 1e-340, ..., 1e-4, 1e4, ..., 1e340)
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

inline uint64_t getIntegerPowTen(int exponent) {
	assert(exponent >= 0 && exponent < 20);
	static const uint64_t powers[] = {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL, 10000000000000ULL, 100000000000000ULL, 1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL, 1000000000000000000ULL, 10000000000000000000ULL};
	return powers[exponent];
}

static const ExplodedFloatingPoint smallPowerCache[] = {
	{1e1, true}, {1e2, true}, {1e3, true}, {1e4, true}, {1e5, true}, {1e6, true}, {1e7, true}, {1e8}
};

inline double raiseToPowTen(uint64_t base, int powTen) {
	static constexpr double powers[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
	if (std::abs(powTen) <= 22 && ((base <= (1ULL<<53)) || ((base & 0xFFF) == 0))) {
		// Base and powTen are exactly representable as doubles
		double d = static_cast<double>(base);
		if (powTen < 0) {
			return d / powers[-powTen];
		}
		return d * powers[powTen];
	}
	if (powTen == 0) {
		return static_cast<double>(base);
	}
	ExplodedFloatingPoint p{base, 0};
	p.normalize(11);
	int biasedPow = powTen + 348;
	int index = biasedPow / 8;
	int shortIndex = (biasedPow % 8) - 1;
	if (index < 0 || index >= static_cast<int>(sizeof(powerCache) / sizeof(powerCache[0]))) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (shortIndex < 0) {
		auto easyMul = p * powerCache[index];
		return easyMul.asDouble();
	}
	auto mul1 = p * smallPowerCache[shortIndex];
	mul1.normalize(11);
	auto mul2 = mul1 * powerCache[index];
	return mul2.asDouble();
}

inline constexpr char digitToAscii(unsigned int d) {
	return '0' + static_cast<char>(d);
}

inline int writeSmallInteger(char* buffer, int integer) {
	if (integer < 0) {
		buffer[0] = '-';
		return 1 + writeSmallInteger(buffer + 1, -integer);
	}
	if (integer >= 100) {
		buffer[0] = digitToAscii(integer / 100);
		integer %= 100;
		buffer[1] = digitToAscii(integer / 10);
		buffer[2] = digitToAscii(integer % 10);
		return 3;
	} else if (integer >= 10) {
		buffer[0] = digitToAscii(integer / 10);
		buffer[1] = digitToAscii(integer % 10);
		return 2;
	} else {
		buffer[0] = digitToAscii(integer);
		return 1;
	}
}

static inline constexpr uint32_t bitCountOf5ToThe(int pow) {
	return ((static_cast<uint32_t>(pow) * 1217359) >> 19) + 1;
}

static inline constexpr uint64_t top32(uint64_t val) {
	return val >> 32;
}

static inline constexpr uint64_t bot32(uint64_t val) {
	return val & 0xFFFFFFFF;
}

static inline void full64BitMultiply(uint64_t m1, uint64_t m2, std::array<uint64_t, 2>& out) {
	uint64_t botL = bot32(m1);
	uint64_t botR = bot32(m2);

	uint64_t t = (botL * botR);
	uint64_t w3 = bot32(t);
	uint64_t k = top32(t);

	m1 = top32(m1);
	t = (m1 * botR) + k;
	k = bot32(t);
	uint64_t w1 = top32(t);

	m2 = top32(m2);
	t = (botL * m2) + k;
	k = top32(t);

	out[0] = (m1 * m2) + w1 + k;
	out[1] = (t << 32) + w3;
}

static inline uint64_t full64x128MultiplyAndShift(uint64_t m1, const std::array<uint64_t, 2>& m2, uint32_t shift) {
	assert(shift > 0);
	assert(shift < 64);
	std::array<uint64_t, 2> rhigh, rlow;
	full64BitMultiply(m1, m2[0], rhigh);
	full64BitMultiply(m1, m2[1], rlow);

	const uint64_t sum = rlow[0] + rhigh[1];
	rhigh[0] += sum < rlow[0]; // overflow
	return (sum >> shift) | (rhigh[0] << (64 - shift));
}

static inline void multiplyAll(const uint64_t& minus, const uint64_t& mid, const uint64_t& plus,
		const std::array<uint64_t, 2>& multiplier, uint32_t shift,
		uint64_t& minusOut, uint64_t& midOut, uint64_t& plusOut) {
	minusOut = full64x128MultiplyAndShift(minus, multiplier, shift);
	midOut = full64x128MultiplyAndShift(mid, multiplier, shift);
	plusOut = full64x128MultiplyAndShift(plus, multiplier, shift);
}

static bool isDivisibleByPowerOf5(uint64_t value, uint32_t power) {
	while (power-- > 0) {
		if ((value % 5) != 0) {
			return false;
		}
		value /= 5;
	}
	return true;
}

static inline void computeShortest(uint64_t minus, uint64_t mid, uint64_t plus, uint32_t exponent, bool even,
		bool minusIsTrailingZeroes, bool midIsTrailingZeroes, uint64_t& out, int32_t& outExponent) {
	outExponent = exponent;
	if (minusIsTrailingZeroes || midIsTrailingZeroes) {
		uint32_t lastRemovedDigit = 0;
		while (plus / 10 > minus / 10) {
			minusIsTrailingZeroes &= (minus % 10) == 0;
			midIsTrailingZeroes &= lastRemovedDigit == 0;
			lastRemovedDigit = mid % 10;
			minus /= 10;
			mid /= 10;
			plus /= 10;
			++outExponent;
		}
		if (minusIsTrailingZeroes) {
			while (minus % 10 == 0) {
				lastRemovedDigit = mid % 10;
				minus /= 10;
				mid /= 10;
				plus /= 10;
				midIsTrailingZeroes &= lastRemovedDigit == 0;
				++outExponent;
			}
			if (midIsTrailingZeroes && lastRemovedDigit == 5 && mid % 2 == 0) {
				lastRemovedDigit = 4;
			}
		}
		out = mid + ((mid == minus && (!even || !minusIsTrailingZeroes)) || lastRemovedDigit >= 5);
		return;
	}

	bool roundUp = false;
	if (plus / 100 > minus / 100) {
		roundUp = (mid % 100) >= 50;
		minus /= 100;
		mid /= 100;
		plus /= 100;
		outExponent += 2;
	}
	while (plus / 10 > minus / 10) {
		roundUp = (mid % 10) >= 5;
		minus /= 10;
		mid /= 10;
		plus /= 10;
		++outExponent;
	}
	out = mid + (mid == minus || roundUp);
}

inline int conformalizeNumberString2(char* buffer, char* integer, int length, int powTen) {
	const int totalPowTen = length + powTen;
	if (totalPowTen <= 19) {
		if (powTen >= 0) {
			std::memcpy(buffer, integer, length);
			// Whole number with no exponent
			// add trailing zeros
			for (int i = length; i < totalPowTen; ++i) {
				buffer[i] = '0';
			}
			return totalPowTen;
		} else if (totalPowTen > 0) {
			// Decimal number with no exponent
			// make room for decimal point and then insert it
			std::memcpy(buffer, integer, totalPowTen);
			buffer[totalPowTen] = '.';
			std::memcpy(buffer + totalPowTen + 1, integer + totalPowTen, length - totalPowTen);
			return length + 1;
		} else if (totalPowTen > -6) {
			// Short decimal < 1
			// Make room for '0.' + any preceding zeros
			const int offset = 2 - totalPowTen;
			buffer[0] = '0';
			buffer[1] = '.';
			for (int i = 2; i < offset; ++i) {
				buffer[i] = '0';
			}
			std::memcpy(buffer + offset, integer, length);
			return offset + length;
		}
	}
	// Use scientific notation
	if (length == 1) {
		buffer[0] = integer[0];
		buffer[1] = 'e';
		return 2 + writeSmallInteger(&buffer[2], powTen);
	} else {
		// make room for conventional decimal and then insert it
		std::memmove(&buffer[2], &buffer[1], length - 1);
		buffer[0] = integer[0];
		buffer[1] = '.';
		std::memcpy(buffer + 2, integer + 1, length - 1);
		buffer[length + 1] = 'e';
		return length + 2 + writeSmallInteger(&buffer[length + 2], totalPowTen - 1);
	}
}

inline int ryu(const double d, char* buffer) {
	if (d == 0.0) {
		buffer[0] = '0';
		return 1;
	}
	if (std::signbit(d)) {
		buffer[0] = '-';
		return 1 + ryu(-d, buffer + 1);
	}
	ExplodedFloatingPoint binary(d);
	bool even = (binary.mantissa & 1) == 0;

	// Shift left so that next highest/lowest floats can be expressed in the same exponent
	uint32_t minusShift = (binary.mantissa != (1ULL << 52)) || (binary.exponent <= 1);
	uint64_t binaryMid = binary.mantissa << 2;
	uint64_t binaryPlus = binaryMid + 2;
	uint64_t binaryMinus = binaryMid - 1 - minusShift;
	int32_t binaryExponent = binary.exponent - 2;

	uint64_t decimalMid, decimalMinus, decimalPlus;
	int32_t decimalExponent;
	bool decimalMidIsTrailingZeros = false, decimalMinusIsTrailingZeros = false;
	if (binaryExponent >= 0) {
		// Calculates floor(binaryExponent * log10(2)), or floor(log10(2^binaryExponent))
		// 78918 / 2^18 approximates log10(2)
		decimalExponent = ((static_cast<uint32_t>(binaryExponent) * 78913) >> 18) - (binaryExponent > 3);
		uint32_t i = decimalExponent - binaryExponent + bitCountOf5ToThe(decimalExponent) - 1 + 125;
		uint32_t shift = i - 64;

		multiplyAll(binaryMinus, binaryMid, binaryPlus, negativePowerTable[decimalExponent], shift,
			decimalMinus, decimalMid, decimalPlus);

		if (decimalExponent <= 21) {
			if (binaryMinus % 5 == 0) {
				decimalMidIsTrailingZeros = isDivisibleByPowerOf5(binaryMid, decimalExponent);
			} else if (even) {
				decimalMinusIsTrailingZeros = isDivisibleByPowerOf5(binaryMinus, decimalExponent);
			} else {
				--decimalPlus;
			}
		}
	} else {
		// binaryExponent < 0
		// Calculates floor(-binaryExponent * log10(5)), or floor(log10(5^(-binaryExponent)))
		// 732923 / 2^20 approximates log10(5)
		uint32_t q = ((static_cast<uint32_t>(-binaryExponent) * 732923) >> 20) - (binaryExponent < -1);
		decimalExponent = q + binaryExponent;
		uint32_t i = -decimalExponent;
		uint32_t b5i = bitCountOf5ToThe(i);
		uint32_t j = q - b5i + 125;
		uint32_t shift = j - 64;

		multiplyAll(binaryMinus, binaryMid, binaryPlus, positivePowerTable[i], shift,
			decimalMinus, decimalMid, decimalPlus);

		if (q <= 1) {
			decimalMidIsTrailingZeros = true;
			if (even) {
				decimalMinusIsTrailingZeros = minusShift;
			} else {
				--decimalPlus;
			}
		} else if (q < 63) {
			decimalMidIsTrailingZeros = (binaryMid & ((1ULL << (q - 1)) - 1)) == 0;
		}
	}

	uint64_t out;
	computeShortest(decimalMinus, decimalMid, decimalPlus, decimalExponent, even,
		decimalMinusIsTrailingZeros, decimalMidIsTrailingZeros, out, decimalExponent);
	char integerBuff[20];
	char* start = writeIntegerToBuff(out, integerBuff + sizeof(integerBuff));
	int32_t length = sizeof(integerBuff) - (start - integerBuff);
	return conformalizeNumberString2(buffer, start, length, decimalExponent);
}

}
}

#endif
