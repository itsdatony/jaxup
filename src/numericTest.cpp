// The MIT License (MIT)
//
// Copyright (c) 2017-2025 Kyle Hawk
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

#include <cinttypes>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <bitset>

#define JAXUP_USE_SHARED_POWER_CACHE
#include <jaxup.h>

using namespace jaxup;

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

int testDouble(double d, jaxup::JsonParser<std::istream>& parser, jaxup::JsonGenerator<std::ostream>& generator, std::stringstream& ss) {
	int error = 0;
	ss.str("");
	ss.clear();
	generator.write(d);
	generator.flush();
	double r = std::strtod(ss.str().c_str(), nullptr);
	if (doubleAsU64(d) != doubleAsU64(r)) {
		std::cout << "Printed string does not recover to value.  Value: " << std::setprecision(17) << d << ", printed: " << ss.str() << ", recovered: " << r << std::endl;
		std::cout << "  Expected:  " << std::bitset<64>(doubleAsU64(d)) << std::endl;
		std::cout << "  Recovered: " << std::bitset<64>(doubleAsU64(r)) << std::endl;
		error |= 2;
	}
	ss.clear();
	JAXUP_TRY {
		parser.nextToken();
	} JAXUP_CATCH (JsonException& e) {
		std::cout << "Parser raised exception: " << e.what() << std::endl;
		error |= 1;
		return error;
	}
	double p = parser.getDoubleValue();
	if (doubleAsU64(d) != doubleAsU64(p)) {
		std::cout << "Roundtrip values do not match.  Expected: " << d << ", got: " << p << std::endl;
		std::cout << "  Expected:  " << std::bitset<64>(doubleAsU64(d)) << std::endl;
		std::cout << "  Evaluated: " << std::bitset<64>(doubleAsU64(p)) << std::endl;
		std::cout << "  Written: " << ss.str() << std::endl;
		error |= 1;
	}

	char buff[200];
	std::snprintf(buff, 200, "%1.16le", d);
	ss.str(buff);
	ss.clear();
	JAXUP_TRY {
		parser.nextToken();
	} JAXUP_CATCH (JsonException& e) {
		std::cout << "Parser raised exception: " << e.what() << std::endl;
		error |= 1;
		return error;
	}
	p = parser.getDoubleValue();
	if (doubleAsU64(d) != doubleAsU64(p)) {
		std::cout << "Values do not match.  Expected: " << d << ", got: " << p << std::endl;
		std::cout << "  Expected:  " << std::bitset<64>(doubleAsU64(d)) << std::endl;
		std::cout << "  Evaluated: " << std::bitset<64>(doubleAsU64(p)) << std::endl;
		error |= 1;
	}
	return error;
}

int testNonStandardFormat(const char* input, double /*expected*/, jaxup::JsonParser<std::istream>& parser, std::stringstream& ss) {
	int error = 0;
	ss.str(input);
	ss.clear();
	parser.nextToken();
	double p = parser.getDoubleValue();
	double expected = std::strtod(input, nullptr);
	if (doubleAsU64(expected) != doubleAsU64(p)) {
		std::cout << "Values do not match.  Expected " << expected << " from input \"" << input << "\", got: " << p << std::endl;
		std::cout << "  Expected:  " << std::bitset<64>(doubleAsU64(expected)) << std::endl;
		std::cout << "  Evaluated: " << std::bitset<64>(doubleAsU64(p)) << std::endl;
		error = 1;
	}
	return error;
}

int main(int /*argc*/, char* /*argv*/[]) {
	std::cout << std::setprecision(17) << std::scientific;
	JsonFactory factory;
	std::stringstream ss;
	auto parser = factory.createJsonParser(ss);
	auto generator = factory.createJsonGenerator(ss);
	double testCases[] = {
		1e23,
		1.123456e23,
		std::numeric_limits<double>::max(),
		std::numeric_limits<double>::min(),
		std::numeric_limits<double>::denorm_min(),
		-65.613616999999977,
		7.2057594037927933e16,
		1.0e-308,
		0.1e-308,
		0.01e-307,
		1.79769e+308,
		2.22507e-308,
		-1.79769e+308,
		-2.22507e-308,
		1e-308,
		0,
		-0.0,
		1.7955348806030474e19,
		1.0154032828453354e19,
		2.267954527701348e60,
		9934509011495037000.0,
		29018956725463772,
		6.0807728793355840e+15,
		1.4752497761390908e+16,
		9.76598962682097729e-162,
		1.72622498213725813e-13,
		5.855406067890361e20,
		1.99442770359396e-309,
		9.905549738666e-282
	};
	int numErrors = 0;
	int numWriteErrors = 0;
	int numReadErrors = 0;
	int numBothErrors = 0;
	for (double d : testCases) {
		int error = testDouble(d, *parser, *generator, ss);
		if (error != 0) {
			++numErrors;
			numWriteErrors += (error & 2) > 0;
			numReadErrors += (error & 1) > 0;
			numBothErrors += error == 3;
			std::cout << "str: " << ss.str() << std::endl;
		};
	}
	std::mt19937_64 mt(123456);
	std::uniform_int_distribution<uint64_t> distribution(0x1, 0x7FEFFFFFFFFFFFFFULL);
	for (unsigned int i = 0; i < 1000000; ++i) {
		uint64_t n = distribution(mt);
		double d = u64AsDouble(n);
		int error = testDouble(d, *parser, *generator, ss);
		if (error != 0) {
			++numErrors;
			numWriteErrors += (error & 2) > 0;
			numReadErrors += (error & 1) > 0;
			numBothErrors += error == 3;
			std::cout << "str: " << ss.str() << std::endl;
		};
	}

	std::pair<const char*, double> nonStandardFormatCases[] = {
		{"1234567890123456780", 1234567890123456768.0},
		{"12.34567890123456780", 12.345678901234567348410564591176807880401611328125},
		{"1.234567890123456780", 1.2345678901234566904321354741114191710948944091796875},
		{"0.0001234567890123456780", 0.000123456789012345671298354066180991139844991266727447509765625},
		{"0.00012345678901234567", 0.000123456789012345671298354066180991139844991266727447509765625},
		{"0.0001234567890123456", 0.00012345678901234558998319112976815858928603120148181915283203125},
		{"0.0001234567890123456E-10", 0.00012345678901234558998319112976815858928603120148181915283203125e-10},
		{"999999999999999999999.99999999", 1e21},
		{"9223372036854775807", 9223372036854775807.0},
		{"9999999999999999999", 1e19},
		{"9999999999999999998", 1e19},
		{"9999999999999999997", 1e19},
		{"9999999999999999996", 1e19},
		{"9999999999999999995", 1e19},
		{"9999999999999999994", 1e19},
		{"999999999999999.9994", 1e15},
		{"0.9999999999999999994", 1.0},
		{"0.009999999999999999994", 1.0e-2},
		{"0.00999999999999999999", 1.0e-2},
		{"9999999999999999994.0000000000", 1e19},
		{"1111111111111111111", 1.1111111111111111e18}
	};
	for (const auto& pair : nonStandardFormatCases) {
		numReadErrors += testNonStandardFormat(pair.first, pair.second, *parser, ss);
	}

	std::cout << "Num double write errors: " << numWriteErrors << std::endl;
	std::cout << "Num double read errors: " << numReadErrors << std::endl;
	std::cout << "Num double both errors: " << numBothErrors << std::endl;

	int64_t intTestCases[] = {
		0, 1, -1, 101, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()
	};

	numWriteErrors = 0;
	char buffer[21];
	buffer[20] = '\0';
	for (const auto& integer : intTestCases) {
		char* start = jaxup::numeric::writeIntegerToBuff(integer, buffer + sizeof(buffer) - 1);
		int64_t read;
		if (std::sscanf(start, "%" SCNd64, &read) != 1 || read != integer) {
			std::cout << "Failed to write: " << integer << std::endl;
			++numWriteErrors;
			++numErrors;
		}
	}

	std::cout << "Num integer write errors: " << numWriteErrors << std::endl;
	std::cout << "Total num errors: " << numErrors << std::endl;

	return numErrors;
}