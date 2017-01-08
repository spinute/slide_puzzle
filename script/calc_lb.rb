fname = ARGV[0]
data = []
open(fname) do |f|
	while l = f.gets
		if l.start_with?('STAT: loop')
			l = f.gets
			data = l.split(',').map(&:to_i)
		elsif l.start_with?('Error:')
			av = data.reduce(:+) / 48.0
			puts "avarage loads: #{av}"
			loads = Array.new(48, 0.0)
			data.each{|e|
				min_i = 0; min = 1234567890;
				loads.each_with_index{|e, i|
					if e < min
						min = e; min_i = i
					end
				}
				loads[min_i] += e.to_f
			}
			p loads.map{|e|e/av.to_f}.max
		else
			;
		end
	end
end
