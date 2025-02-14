#include <stdio.h>

float I [2]; //inputs
float H [2]; //hidden layer neurons before activation
float HS [2]; //hidden layer neurons
float O [1]; //output neuron before activation
float OS [1]; //output neuron 
float L [1]; //loss
float WH [4]; //hidden layer weights {I1->H1, I2->H1, I1->H2, I2->H2}
float WO [2]; //output layer weights
float BH [2]; //hidden layer biases
float BO [1]; //output layer bias


//samples and tests in {in0, in1, out0} format
const float BATCH1 [][3] = {{0,0,0}, {1,0,1}};
const float BATCH2 [][3] = {{1,1,0}, {0,1,1}};
const float TEST [][3] = {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};


void init_weights(unsigned int seed); //initial values between -1.4 and 1.4 (xavier/glorot) 
void forward();
void learn(const float batch[][3], int size, float rate); //run a single epoch and update weights
void test_current_state(const float [][3], int size);

void loss(float x){L[0] = (OS[0] -x)*(OS[0] -x);}
float softsign(float x){return 0.5 * (x / (1 + (x>0 ? x : -x)) + 1);} //shifted and scaled
float softsign_der(float x){return 0.5 / ((1 + (x>0 ? x : -x)) * (1 + (x>0 ? x : -x)));} 



int main(){
  init_weights(42);
  
  //run 10_000 epochs
  for (int i = 0; i<10000; ++i){
    //test_current_state(); 
    //run one epoch, batch whole sample
    learn(BATCH1, sizeof(BATCH1)/sizeof(BATCH1[0]),  0.1);
    learn(BATCH2, sizeof(BATCH2)/sizeof(BATCH2[0]),  0.1);
  }
  test_current_state(TEST, sizeof(TEST)/sizeof(TEST[0]));

}

void forward(){
  H[0] = WH[0] * I[0] + WH[1] * I[1] + BH[0];
  HS[0] = softsign(H[0]);
  H[1] = WH[3] * I[1] + WH[2] * I[0] + BH[1];
  HS[1] = softsign(H[1]);
  O[0] = WO[0] * HS[0] + WO[1] * HS[1] + BO[0];
  OS[0] = softsign(O[0]);
}

void learn(const float batch[][3], int size, float rate){
  //average gradients of weights and biases
  float GBO [1] = {0}, GWO [2] = {0}, GBH [2] = {0}, GWH [4] = {0};

  //run gradient calculations (hardcoded equations, derived by hand)
  for (int i = 0; i<size; ++i){
    I[0] = batch[i][0]; I[1] = batch[i][1];
    forward();
    loss(batch[i][2]);

    GBO[0] += 2*(OS[0] - batch[i][2])*softsign_der(O[0]);
    for(int j = 0; j<2; ++j)
      GWO[j] += HS[j] * 2 * (OS[0] - batch[i][2]) * softsign_der(O[0]) /size;
    for(int j = 0; j<2; ++j)
      GBH[j] += softsign_der(H[j]) * WO[j] * 2 * (OS[0] - batch[i][2]) * softsign_der(O[0]) /size;
    for(int j = 0; j<2; ++j)
      GWH[j] +=  softsign_der(H[0]) * WO[0] * 2 * (OS[0] - batch[i][2]) * softsign_der(O[0]) * I[j] /size;
    for(int j = 0; j<2; ++j)
      GWH[j+2] += softsign_der(H[1]) * WO[1] * 2 * (OS[0] - batch[i][2]) * softsign_der(O[0]) * I[j] /size;
  }

  //update weights and biases
  BO[0] -= rate * GBO[0];
  for(int j = 0; j<2; ++j)
    WO[j] -= rate * GWO[j];
  for(int j = 0; j<2; ++j)
    BH[j] -= rate * GBH[j];
  for(int j = 0; j<4; ++j)
    WH[j] -= rate * GWH[j];

}

void init_weights(unsigned int seed){
  float rand_f[15];
  for (int i = 0; i < 10; ++i){
    //simple RNG for range -1.4 to 1.4
    seed = seed ^ (seed << 3); seed = seed ^ (seed >> 2); seed = seed ^ (seed << 1);
    rand_f[i] = (float)(seed % 100) / 35 - 1.4;
  }
  float* rand_f_it = rand_f - 1;
  for(int i = 0; i<4; ++i) WH[i] = *(++rand_f_it);
  for(int i = 0; i<2; ++i) WO[i] = *(++rand_f_it);
  for(int i = 0; i<2; ++i) BH[i] = *(++rand_f_it);
  BO[0] = *(++rand_f_it);
}

void test_current_state(const float test[][3], int size){
  float agg_loss = 0; //aggregate loss
  printf("EXPECTED: {%f", test[0][2]);
  for(int i = 1; i<size; ++i)
    printf(", %f", test[i][2]);

  I[0] = test[0][0]; I[1] = test[0][1];
  forward(); loss(test[0][2]); agg_loss += L[0];
  printf("}, PREDICTED: {%f", OS[0]);
  for(int i = 1; i<size; ++i){
    I[0] = test[i][0]; I[1] = test[i][1];
    forward(); loss(test[i][2]); agg_loss += L[0];
    printf(", %f", OS[0]);
  }
  printf("}, LOSS: %.10f\n", agg_loss/size);
}
