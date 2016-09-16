require 'rake/clean'
require 'pathname'

CC = "clang"
OPT = "-O2 -Wall -Wextra -m64 -g"
CFLAGS = ENV['CFLAGS']

TEST = "src/test"
UNITY = "unity/src"
UNITY_FIXTURE = "unity/extras/fixture/src"

directory "bin"

CLEAN.include('src/*.o')
CLOBBER.include('bin/*')

task :default => [:construct, :main, :test] do
end

SRC = FileList["src/*.c"]
OBJ = SRC.ext('o')

SRC_WOMAIN = SRC.exclude("src/main.c")
OBJ_WOMAIN = SRC_WOMAIN.ext('o')
TESTSRC = FileList["#{TEST}/*.c", "#{UNITY}/*.c", "#{UNITY_FIXTURE}/*.c"]
TESTOBJ = TESTSRC.ext('o')

desc 'main'
task :main => OBJ do |t|
	sh "#{CC} #{t.prerequisites.join ' '} -o bin/#{t.name} #{OPT} #{CFLAGS} -Isrc"
end

desc 'test'
task :test => TESTOBJ + OBJ_WOMAIN do |t|
	sh "#{CC} #{t.prerequisites.join ' '} -o bin/#{t.name} #{OPT} #{CFLAGS} -Isrc -I#{UNITY} -I#{UNITY_FIXTURE}"
	sh "bin/test"
end

desc 'dir setup'
task :construct => [:bin] do
end

rule '.o' => '.c' do |t|
	sh "#{CC} -c #{t.source} -o #{t.name} #{OPT} #{CFLAGS} -Isrc -I#{UNITY} -I#{UNITY_FIXTURE}"
end

desc 'clang-format'
task :fmt do
	sh "clang-format -i src/*.[ch]"
	sh "clang-format -i src/test/*.[ch]"
end

desc 'cscope and ctags'
task :tags do
	sh "ctags -R ."
	sh "cscope -bR"
end
