def fmt(w, h, positions)
return <<EOS
#{h} #{w}
starting positions for each tile:
#{positions.join "\n"}
goal positions:
#{(0..(w*h-1)).to_a.join("\n")}
EOS
end

def tiles2positions(w, h, tiles)
	Array.new(w*h){|i| tiles.index i.to_s}
end

def korf2burns
	w = h = 4
	100.times do |i|
		tiles = open("benchmarks/korf/prob%03d" % i).readline.strip.split(' ')
		positions = tiles2positions w, h, tiles
		content = fmt w, h, positions
		File.write("benchmarks/burns/prob%03d" % i, content)
	end
end

def korf2burns25
	w = h = 5
	1.upto 50 do |i|
		tiles = open("benchmarks/all25/%03d" % i).readline.strip.split(' ')
		positions = tiles2positions w, h, tiles
		content = fmt w, h, positions
		File.write("benchmarks/burns_25/%03d" % i, content)
	end
end

def rand25burns
	w = h = 5
	50.step(500, 50) do |n|
		10.times do |i|
			fname = "#{n}_#{i}"
			tiles = open("benchmarks/rand25/#{fname}").readline.strip.split(' ')
			positions = tiles2positions w, h, tiles
			content = fmt w, h, positions
			File.write("benchmarks/burns_rand25/#{fname}", content)
		end
	end
end

korf2burns25()
