WIDTH = 4
N_DIR = 4
DIR = ["RIGHT", "LEFT", "DOWN", "UP"]

print "{"
WIDTH.times do |x|
	WIDTH.times do |y|
		# right
		print "#{x < WIDTH-1},"
		# left
		print "#{x > 0},"
		# down
		print "#{x < WIDTH-1},"
		# up
		print "#{x > 0},"
		print " "
	end
	puts ""
end
puts "};"
