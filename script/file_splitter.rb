print "file name:"
fname = gets.chop
cnt = 0

File.open(fname).each do |l|
	File.write "benchmarks/korf/prob%03d" % cnt, l
	cnt += 1
end
