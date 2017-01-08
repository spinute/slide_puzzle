fname = ARGV[0]
data = []
open(fname) do |f|
	while l = f.gets
		if l.start_with?('STAT: efficiency')
			l = f.gets
			data = l.split(',').map(&:to_i)
		elsif l.start_with?('Error:')
			h = data.reduce(Hash.new){|h, e| h[e] = h[e].nil? ? 1 : h[e]+1; h}
			0.upto(32){|i| print "#{h[i] || 0}, "}
			puts ''
		else
			;
		end
	end
end
