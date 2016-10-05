#!/Users/spinute/build/bin/ruby

require 'benchmark'

100.times do |i|
	puts "prob%03d" % i
	puts Benchmark.measure {
		`bin/main -i benchmarks/korf/prob#{"%03d" % i} -s 10`
	}
end
