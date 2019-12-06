#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main(int argc, char** argv) 
{
   if (argc == 1)
   {
      printf("usage: %s <model_dir>\n", argv[0]);
      return -1;
   }

   TF_Graph* graph = TF_NewGraph();
   TF_SessionOptions* options = TF_NewSessionOptions();
   TF_Status* status = TF_NewStatus();

   TF_Buffer* run = TF_NewBuffer();
   TF_Buffer* meta = TF_NewBuffer();

   // default tags
   const char* tags[1] = {"serve"};

   // load session
   TF_Session* tfs = TF_LoadSessionFromSavedModel(
      options, 
      run, // NULL, // const TF_Buffer* run_options,
      argv[1], 
      tags, 
      1,
      graph, 
      meta, // NULL, //TF_Buffer* meta_graph_def, 
      status
   );

   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
      return -1;
   }

   // list operations
   size_t pos = 0;
   TF_Operation* oper;
   while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) 
   {
      TF_Output tensor;
      const char* name = TF_OperationName(oper);
      tensor.oper = TF_GraphOperationByName(graph, name);
      tensor.index = 0;
      int n_ins = TF_OperationNumInputs(oper);
      int n_outs = TF_OperationNumOutputs(oper);
      int n_dims = -1;
      if (n_ins != 0 && n_outs !=0)
      {
         n_dims = TF_GraphGetTensorNumDims(graph, tensor, status);
         if (TF_GetCode(status) != TF_OK)
         {
            printf("Error: Tensorflow: %s\n", TF_Message(status));
            return -1;
         }
      }
      printf("op: name=%s ins=%d outs=%d dims=%d", name, n_ins, n_outs, n_dims);
      if (n_dims != -1)
      {
         int64_t dims[1024];
         TF_GraphGetTensorShape(graph, tensor, dims, n_dims, status);
         if (TF_GetCode(status) != TF_OK)
         {
            printf("Error: Tensorflow: %s\n", TF_Message(status));
            return -1;
         }
         printf(" shape=[");
         for (int i=0; i<n_dims; i++)
         {
            printf("%ld%s", dims[i], (i<n_dims-1) ? "," : "");
         }
         printf("]\n");
      }
      else
      {
         printf(" shape=?\n");
      }
   }

   // terminate
   TF_CloseSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
      return -1;
   }
   TF_DeleteSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
      return -1;
   }
   TF_DeleteBuffer(meta);
   TF_DeleteBuffer(run);
   TF_DeleteSessionOptions(options);
   TF_DeleteGraph(graph);
   TF_DeleteStatus(status);

   return 0;
}

