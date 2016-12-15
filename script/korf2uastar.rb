def korf2uastar
	w = h = 4
	100.times do |i|
		tiles = open("benchmarks/all/prob%03d" % i).readline.strip.split(' ')
		output = tiles.map{|i| i=='0' ? 0 : w*h-i.to_i}
		content =  output.join " "
		File.write("benchmarks/uastar/prob%03d" % i, content)
	end
end

korf2uastar()
