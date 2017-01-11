#!/Users/spinute/build/bin/ruby

require 'benchmark'
require 'timeout'

#100.times do |i|
[1, 5, 13, 25, 30, 32, 37, 38, 40, 44].each do |i|
	puts "prob%03d" % i
		puts Benchmark.measure {
			#puts `bin/cpumain benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/cubase benchmarks/all/prob#{"%03d" % i}`
			puts `bin/cupdb benchmarks/all25/#{"%03d" % i}`
			#puts `bin/cusingle benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/cumain benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/cumem benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/15md_solver idastar < benchmarks/burns/prob#{"%03d" % i}`
		}
end
