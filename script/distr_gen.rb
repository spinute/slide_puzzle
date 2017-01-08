fname = ARGV[0]
open(fname) do |f|
	while l = f.gets
		if l.start_with?('1','2','3','4','5','6','7','8','9','0')
			data = l.split(',').map(&:to_i)
			av = data.reduce(:+) / data.length
			puts "avarage loads: #{av}"
			p data.map{|e| e/av}.reduce(Hash.new){|h, e| h[e] = h[e].nil? ? 1 : h[e]+1; h}
		else
			puts l
		end
	end
end
