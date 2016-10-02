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

ar = (0..15).to_a
pos = 0

if ARGV[0].nil?
	raise "argnum is required"
end
st = ARGV[0].to_i
puts "#{st} step random walk"

st.times do
	begin
		dir = rand_dir
	end until movable pos, dir

	pos = move ar, pos, dir
end

problem_dump ar
