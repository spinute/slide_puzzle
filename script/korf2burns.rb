def fmt(positions)
return <<EOS
4 4
starting positions for each tile:
#{positions.join "\n"}
goal positions:
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
EOS
end

def tiles2positions(tiles)
	Array.new(16){|i| tiles.index i.to_s}
end

100.times do |i|
	tiles = open("benchmarks/korf/prob%03d" % i).readline.strip.split(' ')
	positions = tiles2positions tiles
	content = fmt positions
	File.write("benchmarks/burns/prob%03d" % i, content)
end
