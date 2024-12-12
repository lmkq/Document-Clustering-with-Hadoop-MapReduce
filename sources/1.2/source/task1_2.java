import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
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
import java.util.ArrayList;


public class task1_2 {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text>{
    private Text termId = new Text();
    private Text pairVal = new Text();
    private int row=1;
    
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if(row<2) {
            row++;
        }
        else {
        String [] cont = value.toString().split("\\s+");
        termId.set(cont[0]);
        pairVal.set(cont[1]+" "+cont[2]);
        context.write(termId,pairVal);
        }
    }
  }
  public static class IntSumReducer
       extends Reducer<Text,Text,Text,Text> {
    private Text result = new Text();
    
    public void reduce(Text key, Iterable<Text> values, Context context ) 
    throws IOException, InterruptedException {
    	
        ArrayList<String> PairVal = new ArrayList<>(); 
        Integer sum = 0;
      for (Text val : values) {
        String strVal = val.toString(); 
        PairVal.add(strVal); 
        String ValCon[] = strVal.split("\\s+");
        Integer Valadd = Integer.parseInt(ValCon[1]);
        sum = sum + Valadd;
      }
      if (sum>=3){
    	  for (String val : PairVal) {
              result.set(val);
              context.write(key, result);
          }
      }
    }
  }

  public static HashMap<String,Integer> termId = new HashMap<>();
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
	        Path MTXFile = new Path(MTXDir + "/OutputTask1_2.mtx");
	        
	        List<String> MTXContent = new ArrayList<>();
            FileStatus[] Filestate = Filesys.globStatus(new Path(MTXDir + "/part-r-*"));
            for (FileStatus state : Filestate) {
                Path statePath = state.getPath();
                try (FSDataInputStream InStream = Filesys.open(statePath);
                     BufferedReader Buffread = new BufferedReader(new InputStreamReader(InStream))) {
                    String line;
                    while ((line = Buffread.readLine()) != null) {
                        String[] cont = line.split("\\s+");
                        Integer val1 = Integer.parseInt(cont[0]);
                        Integer val2 = Integer.parseInt(cont[1]);
                        Integer val3 = Integer.parseInt(cont[2]);
                        MTXContent.add(String.format("%d %d %d", val1, val2, val3));
                    }
                }
            }
            
	        try (FSDataOutputStream OutStream = Filesys.create(MTXFile);
	             BufferedWriter Buffwrite = new BufferedWriter(new OutputStreamWriter(OutStream))) {
	        	
	            MTXContent.sort(Comparator.naturalOrder());
	            Buffwrite.write("%%MatrixMarket matrix coordinate real general\n");
	            Buffwrite.write(String.format("%d %d %d%n", termId.size(), docId.size(),MTXContent.size()));
	            for (String x : MTXContent) {
	                Buffwrite.write(x + "\n");
	            }
	        } finally {
	            Filesys.close();
	        } 
	    }
	}

  
  public static void main(String[] args) throws Exception {
    loadDocId("./input/bbc.docs");
    loadTermId("./input/bbc.terms");

    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "task1_2");
    job.setJarByClass(task1_2.class);

    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    
    FileInputFormat.addInputPath(job, new Path(args[0]));
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
