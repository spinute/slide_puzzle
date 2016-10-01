#table[who, where, dir] = h_value_diff

DIR = ["RIGHT", "LEFT", "DOWN", "UP"]
WIDTH = 4

print "{"
0.upto 15 do |i|
	0.upto 15 do |from|
		0.upto 3 do |dir|
			right_x = i % WIDTH
			right_y = i / WIDTH

			from_x = from % WIDTH
			from_y = from / WIDTH

			if DIR[dir] == "RIGHT"
				print "#{right_x > from_x ? 1 : -1}, "
			elsif DIR[dir] == "LEFT"
				print "#{right_x < from_x ? 1 : -1}, "
			elsif DIR[dir] == "DOWN"
				print "#{right_y < from_y ? 1 : -1 }, "
			elsif DIR[dir] == "UP"
				print "#{right_y > from_y ? 1 : -1}, "
			end
		end
	end
	puts ""
end
puts "};"
