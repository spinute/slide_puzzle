#!/Users/spinute/build/bin/ruby

require 'benchmark'

100.times do |i|
	puts "prob%03d" % i
	$stderr.print i%10==0 ? '*' : '.'
	puts Benchmark.measure {
		`./cpumain benchmarks/all/prob#{"%03d" % i}`
	}
end
