#table[who, where, dir] = h_value_diff

DIR = [:up, :right, :left, :down]
WIDTH = 4

puts "/* usage: h_diff[who*64+from*4+dir] */"
print "static int h_diff_table[STATE_N*STATE_N*N_DIR] = {"
0.upto 15 do |i|
	0.upto 15 do |from|
		0.upto 3 do |dir|
			right_x = i % WIDTH
			right_y = i / WIDTH

			from_x = from % WIDTH
			from_y = from / WIDTH

			case DIR[dir]
			when :left
				print "#{right_x < from_x ? -1 : 1}, "
			when :right
				print "#{right_x > from_x ? -1 : 1}, "
			when :up
				print "#{right_y < from_y ? -1 : 1}, "
			when :down
				print "#{right_y > from_y ? -1 : 1}, "
			end
		end
	end
	puts ""
end
puts "};"
