import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.io.InputStreamReader;
import java.util.Comparator;
import java.util.regex.Matcher;
import java.util.ArrayList;
import java.util.regex.Pattern;

import javax.naming.Context;


public class task1_1 {
  public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

      String PreClean;
      String fileInput = ((FileSplit) context.getInputSplit()).getPath().toString();
      String[] fileItem = fileInput.split("/");
      int len = fileItem.length;
      String fileTail = fileItem[len-1].split("\\.")[0];
            if(fileTail.equals("README")) { return;}
      
      String fileId = fileItem[len-2]+"."+fileTail;
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        PreClean = itr.nextToken();
        String Cleantext = CleanTheText(PreClean);
        if (!stopWords.contains(Cleantext) && Cleantext.length() > 0){
            int checkTermNumber = HandleWord(Cleantext);
            if(checkTermNumber !=0){
              word.set(checkTermNumber + " " + fileId);
              context.write(word, one);
            }
        } 
      }
    }
  }
public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
  private IntWritable result = new IntWritable();
  public void reduce(Text key, Iterable<IntWritable> values,Context context) throws IOException, InterruptedException {
    int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
    result.set(sum);
    context.write(key, result);
  }
}
  public static String CleanTheText(String PreClean){
    String specialChar = "~!@#$%^&*()\\-+\\[\\]\"':.,<>";
    Pattern Regex = Pattern.compile("[" + Pattern.quote(specialChar) + "]");
    Matcher Detectmatch = Regex.matcher(PreClean);
    String Cleantext = (Detectmatch.replaceAll("")).toLowerCase();
    return Cleantext;
  }
  public static int HandleWord(String Cleantext){
    Integer posTerm = termId.get(Cleantext);
      if(posTerm != null){ return posTerm;}
    return 0;
  }
  

  public static HashMap<String,Integer> termId = new HashMap<>();
  private static Set<String> stopWords = new HashSet<>();
  public static HashMap<String, Integer> docId = new HashMap<>();
  public static void loadTermId(String TermIdfilePath) throws IOException{
    int Line=1;
    try (BufferedReader br = new BufferedReader(new FileReader(TermIdfilePath))) {
        String word;
        while ((word = br.readLine()) != null) {
            termId.put(word,Line);
            Line++;
        }
    }
  } 
    public static void loadStopWords(String StopfilePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(StopfilePath))) {
            String word;
            while ((word = br.readLine()) != null) {
                stopWords.add(word.trim());
            }
        }
    }
  public static void loadDocId(String DocIdFilePath) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(DocIdFilePath))) {
        String word;
        int Line = 1;
        while ((word = br.readLine()) != null) {
            docId.put(word, Line);
            Line++;
        }
      }
    } 
  public static class ConvertMTXFile {
	    public static void MTXConverter(String MTXDir) throws IOException {
	        Configuration conf = new Configuration();
	        FileSystem Filesys = FileSystem.get(conf);
	        Path MTXFile = new Path(MTXDir + "/OutputTask1_1.mtx");
	        
	        List<String> MTXContent = new ArrayList<>();
          FileStatus[] Filestate = Filesys.globStatus(new Path(MTXDir + "/part-r-*"));
            for (FileStatus state : Filestate) {
                Path statePath = state.getPath();
                try (FSDataInputStream InStream = Filesys.open(statePath);
                     BufferedReader Buffread = new BufferedReader(new InputStreamReader(InStream))) {
                    String line;
                    while ((line = Buffread.readLine()) != null) {
                        String[] cont = line.split("\\s+");
                        MTXContent.add(String.format("%d %d %d", Integer.parseInt(cont[0]), docId.get(cont[1]), Integer.parseInt(cont[2])));
                    }
                }
            }
	        
	        try (FSDataOutputStream OutStream = Filesys.create(MTXFile);
	             BufferedWriter Buffwrite = new BufferedWriter(new OutputStreamWriter(OutStream))) {
	            MTXContent.sort(Comparator.naturalOrder());
	            Buffwrite.write("%%MatrixMarket matrix coordinate real general\n");
	            Buffwrite.write(String.format("%d %d %d%n", termId.size(), docId.size(), MTXContent.size()));
	            for (String x : MTXContent) {
	                Buffwrite.write(x + "\n");
	            }
	        } finally {
	            Filesys.close();
	        }
	    }
	}

  
  public static void main(String[] args) throws Exception {
    loadStopWords("./input/stopwords.txt");
    loadDocId("./input/bbc.docs");
    loadTermId("./input/bbc.terms");

    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "task1_1");
    job.setJarByClass(task1_1.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileInputFormat.setInputDirRecursive(job, true);
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
  //System.exit(job.waitForCompletion(true) ? 0 : 1);
    if (job.waitForCompletion(true)) {
    	ConvertMTXFile.MTXConverter(args[1]);
        System.out.println("DONE");
      System.exit(0);
    } else {
      System.exit(1);
    }
  }
}
