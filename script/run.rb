#!/Users/spinute/build/bin/ruby

require 'benchmark'
require 'timeout'

100.times do |i|
	puts "prob%03d" % i
	$stderr.print i%10==0 ? '*' : '.'
		puts Benchmark.measure {
			#puts `bin/cpumain benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/cusingle benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/cumain benchmarks/all/prob#{"%03d" % i}`
			puts `bin/cumem benchmarks/all/prob#{"%03d" % i}`
			#puts `bin/15md_solver idastar < benchmarks/burns/prob#{"%03d" % i}`
		}
end
