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
		return writeUnsignedIntegerToBuff(static_cast<uint64_t>(value), endMarker);
	} else {
		char* start = writeUnsignedIntegerToBuff(0 - static_cast<uint64_t>(value), endMarker);
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

	double asDouble(bool exact) const {
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
			exact = exact & ((temp.mantissa & 0x3FF) == 0);
			out += (out & 1) && ((out & 2) || !exact);
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
			exact = exact & ((temp.mantissa & (0xFFFFFFFFFFFFFFFFULL >> (53 + biasedExponent))) == 0);
			out += (out & 1) && ((out & 2) || !exact);
			// overflow happily rolls into a normalized number, so no need to check
			out = out >> 1;
		}
		return u64AsDouble(out);
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

inline uint64_t getIntegerPowTen(int exponent) {
	assert(exponent >= 0 && exponent < 20);
	static const uint64_t powers[] = {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL, 10000000000000ULL, 100000000000000ULL, 1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL, 1000000000000000000ULL, 10000000000000000000ULL};
	return powers[exponent];
}

inline uint32_t ilog2(uint64_t value, uint32_t numDigits) {
	// There are certainly faster ways to do this, but this is simple and portable
	// 1: Convert floor(log10(value)) to corresponding value in log2
	const uint32_t log2 = ((numDigits - 1) << 20) / 315653;
	// 2: Divide value by 2^guess and use a lookup table for whatever's left
	// Quotient really shouldn't go past 11, but add a few more just in case
	static const uint32_t map[] = {0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
	return log2 + map[value >> log2];
}

static inline constexpr uint32_t bitCountOf5ToThe(int pow) {
	return ((static_cast<uint32_t>(pow) * 1217359) >> 19) + 1;
}

static inline constexpr int32_t sgn(int32_t v) {
	return (0 < v) - (v < 0);
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

static inline bool isDivisibleByPowerOf2(const uint64_t value, const uint32_t power) {
	return (value & ((1ULL << power) - 1)) == 0;
}

inline double raiseToPowTen(uint64_t base, int powTen, int numDigits) {
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
	if (base == 0 || (powTen + numDigits) <= -324) {
		return 0.0;
	}
	if (powTen + numDigits >= 310) {
		return std::numeric_limits<double>::infinity();
	}
	if (numDigits > 17) {
		const uint32_t surplus = numDigits - 17;
		uint32_t divisor = static_cast<uint32_t>(getIntegerPowTen(surplus));
		uint32_t remainder = base % divisor;
		base /= divisor;
		uint32_t half = divisor >> 1;
		base += (remainder > half) || (remainder == half && (base & 1) == 1);
		powTen += surplus;
		numDigits = 17 + (base == 100000000000000000ULL);
	}
	const uint32_t log2 = ilog2(base, numDigits);
	const uint32_t shift = log2 - 1 - 53 + 61;
	assert(shift < 64);
	const uint32_t absPowTen = std::abs(powTen);
	static const std::array<uint64_t, 2>* tables[] = {negativePowerTable, positivePowerTable};
	const auto& factor = tables[powTen > 0][absPowTen];

	ExplodedFloatingPoint p;
	p.exponent = shift + powTen + sgn(powTen) * bitCountOf5ToThe(absPowTen) + (powTen < 0) - 61;
	p.mantissa = full64x128MultiplyAndShift(base, factor, shift);
	int powDiff = p.exponent - powTen;
	bool exact = (powTen < 0 && isDivisibleByPowerOf5(base, -powTen)) ||
		(powTen >= 0 && (powDiff < 0 || (powDiff < 64 && isDivisibleByPowerOf2(base, powDiff))));
	return p.asDouble(exact);
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

inline int conformalizeNumberString(char* buffer, char* integer, int length, int powTen) {
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
	if (std::signbit(d)) {
		buffer[0] = '-';
		return 1 + ryu(-d, buffer + 1);
	}
	if (d == 0.0) {
		buffer[0] = '0';
		return 1;
	}
	ExplodedFloatingPoint binary(d);
	bool even = (binary.mantissa & 1) == 0;

	// Shift left so that next highest/lowest floats can be expressed in the same exponent
	bool minusShift = (binary.mantissa != (1ULL << 52)) || (binary.exponent <= 1);
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
	int length = static_cast<int>(sizeof(integerBuff) - (start - integerBuff));
	return conformalizeNumberString(buffer, start, length, decimalExponent);
}

}
}

#endif
