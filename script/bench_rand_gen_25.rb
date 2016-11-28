50.step(500, 50) do |n|
	10.times do |i|
		`ruby script/rand_prob_generator.rb #{n} > benchmarks/rand25/#{n}_#{i}`
	end
end
