require 'rake/clean'
require 'pathname'

CC = "clang"
OPT = "-O2 -Wall -Wextra -m64 -g"
CFLAGS = ENV['CFLAGS']

INCLUDE = "src"

directory "bin"

CLEAN.include('src/*.o')
CLOBBER.include('bin/*')

desc 'all setup'
task :default => [:construct, :main] do
end

SRC = FileList["src/*.c"]
OBJ = SRC.ext('o')

desc 'main'
task :main => OBJ do |t|
	sh "#{CC} #{t.prerequisites.join ' '} -o bin/#{t.name} #{OPT} #{CFLAGS} -I#{INCLUDE}"
end

desc 'dir setup'
task :construct => [:bin] do
end

rule '.o' => '.c' do |t|
	sh "#{CC} -c #{t.source} -o #{t.name} #{OPT} #{CFLAGS} -I#{INCLUDE}"
end

desc 'clang-format'
task :fmt do
	sh "clang-format -i src/*.[ch]"
end

desc 'cscope and ctags'
task :tags do
	sh "ctags -R ."
	sh "cscope -bR"
end
