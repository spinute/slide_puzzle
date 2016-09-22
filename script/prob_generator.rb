def problem_dump(ar)
	ar.each{|e| print "#{e} "}
	puts ""
end

def calc_array_parity(ar)
	ar = ar.dup
	cnt = 0
	for i in 0...ar.length do
		for j in (i+1)...ar.length do
			if i+1 == ar[j]
				ar[i], ar[j] = ar[j], ar[i]
				cnt += 1
			end
		end
	end

	cnt % 2
end

def calc_goal_parity(ar)
	w = Math.sqrt(ar.length).to_i
	i = ar.find_index{|e| e==0}
	((w-1 - i/w) + (w-1 - i%w))%2
end

def solution_exist?(ar)
	calc_array_parity(ar) == calc_goal_parity(ar)
end

# goal is assumed to be [1,2,3,...,N-1,0]. 0 is an empty tile
w= 3
n = w**2
(0..(n-1)).to_a.permutation {|ar| problem_dump ar if solution_exist? ar}
