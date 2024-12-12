package com.hadoop.mapreduce;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.Text;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
public class UniqueWords {
    public static class UniqueWordsMapper extends Mapper<Object, Text, Text, IntWritable> {
        @Override
        public void map(Object key, Text value, Context con) throws IOException, InterruptedException {
            String[] content = value.toString().split("\\s++");
            con.write(new Text(content[0]), new IntWritable(1));
        }
    }

    public static class UniqueWordsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> value, Context con) throws IOException, InterruptedException {
            con.getCounter(WordCounter.wordcounter.counter).increment(1);
            con.write(key, new IntWritable(1));
        }
    }
}
