#!/Users/spinute/build/bin/ruby

require 'benchmark'

100.times do |i|
	puts Benchmark.measure {
		`bin/main -i benchmarks/korf/prob#{"%03d" % i} -s 1`
	}
end
