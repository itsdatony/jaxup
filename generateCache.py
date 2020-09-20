#!/usr/bin/env python

from decimal import *
import math

getcontext().prec = 300

def get_multiplier(i):
	p = Decimal(5) ** i
	#j = floor(log2(p))
	j = p.log10() // Decimal(2).log10()
	#m = floor(5^i/2^(j-124))
	m = p // (Decimal(2) ** (j - 124))
	return m

def get_inverse_multiplier(i):
	p = Decimal(5) ** i
	#j = floor(log2(p))
	j = p.log10() // Decimal(2).log10()
	m = ((Decimal(2) ** (j + 125)) // p) + 1
	#m = floor(2^(j+125)/5^i) + 1
	return m

def write_val(v, out):
	split = Decimal(2) ** 64
	top = v // split
	bottom = v % split
	out.write('\t{{{}ULL, {}ULL}},\n'.format(top,bottom))

with open('include/jaxup_power_tables.h', 'w') as out:
	with open('License.md') as license:
		for line in license:
			out.write('// ' + line)
	out.write('\n#ifndef JAXUP_POWER_TABLES_H\n')
	out.write('#define JAXUP_POWER_TABLES_H\n\n')
	out.write('#include <array>\n\n')
	out.write('namespace jaxup {\n')
	out.write('namespace numeric {\n\n')
	out.write('static const std::array<uint64_t, 2> positivePowerTable[] = {\n')
	for i in range(0, 326):
		write_val(get_multiplier(i), out)
	out.write('};\n\n')
	out.write('static const std::array<uint64_t, 2> negativePowerTable[] = {\n')
	for i in range(0, 342):
		write_val(get_inverse_multiplier(i), out)
	out.write('};\n\n')

	out.write('}\n}\n\n#endif\n')
