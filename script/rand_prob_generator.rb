# goal is assumed to be
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 0

$N = 4
$DIR = [:right, :left, :up, :down]

def problem_dump(ar)
	ar.each{|e| print "#{e} "}
	puts ""
end

def movable(pos, dir)
	case dir
	when :right
		pos % $N < $N-1
	when :left
		pos % $N > 0
	when :up
		pos / $N > 0
	when :down
		pos / $N < $N-1
	end
end

def move(ar, pos, dir)
	new_pos = {:right => pos+1, :left => pos-1, :up => pos-$N, :down => pos+$N}[dir]
	ar[pos], ar[new_pos] = ar[new_pos], ar[pos]
	new_pos
end

def rand_dir
	$DIR.sample
end

ar = (1..15).to_a << 0
pos = 15

if ARGV[0].nil?
	raise "argnum is required"
end
st = ARGV[0].to_i

st.times do
	begin
		dir = rand_dir
	end until movable pos, dir

	pos = move ar, pos, dir
end

problem_dump ar
