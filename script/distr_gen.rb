fname = ARGV[0]
max_k = 0
av_prev = 0
open(fname) do |f|
	while l = f.gets
		if l.start_with?('1','2','3','4','5','6','7','8','9','0')
			data = l.split(',').map(&:to_i)
			av = data.reduce(:+) / data.length
			puts "max_loads: #{max_k}" if av_prev > av*5
			puts "avarage loads: #{av}"
			av_prev = av
			distr = data.map{|e| e/av}.reduce(Hash.new){|h, e| h[e] = h[e].nil? ? 1 : h[e]+1; h}
			max_k = 0
			distr.each{|k, v| max_k = k if max_k < k}
		else
			;#puts l
		end
	end
	puts "max_loads: #{max_k}"
end
