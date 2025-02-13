/*
simplest implementation of
2 inputs, 1 hidden layer with 2 neurons, 1 output
*/
#include <stdio.h>

float I [2]; //INPUTS
float H [2]; //HIDDEN LAYER NEURONS BEFORE ACTIVATION
float HS [2]; //HIDDEN LAYER NEURONS
float O [1]; //OUTPUT NEURON BEFORE ACTIVATION
float OS [1]; //OUTPUT NEURON
float L [1]; //LOSS
float WH [4]; //HIDDEN LAYER WEIGHTS {I1->H1, I2->H2, I1->H1, I2->H2}
float WO [2]; //OUTPUT LAYER WEIGHTS
float BH [2]; //HIDDEN LAYER BIASES
float BO [1]; //OUTPUT LAYER BIAS

//const float BATCH [4][3] = {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,1}}; //OR
//const float BATCH [4][3] = {{0,0,0}, {0,1,0}, {1,0,0}, {1,1,1}}; //AND
//const float BATCH [4][3] = {{0,0,1}, {0,1,0}, {1,0,0}, {1,1,0}}; //NOR
//const float BATCH [4][3] = {{0,0,1}, {0,1,1}, {1,0,1}, {1,1,0}}; //NAND
const float BATCH [4][3] = {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,1}}; //XOR

void U (unsigned int seed, float *u); //fills "u" with numbers from -1.4 to 1.4 (xavier/glorot)
void forward();
void loss(float x); //return squared error between OS[0] and x
void learn(const float batch[4][3], float rate);
float softsign(float x){
  return 0.5 * (x / (1 + (x>0 ? x : -x)) + 1);
} //shifted and scaled softsign (0->1)
float softsign_der(float x){
  return 0.5 / ((1 + (x>0 ? x : -x))*(1 + (x>0 ? x : -x)));
} //modified softsign derivative



int main(){
  unsigned int seed = 42;

  //initialize weights and biases with U(-1.4, 1.4)
  float rand_f [15];
  U(seed, rand_f);
  float* rand_f_it = rand_f - 1;
  for(int i = 0; i<4; ++i) WH[i] = *(++rand_f_it);
  for(int i = 0; i<2; ++i) WO[i] = *(++rand_f_it);
  for(int i = 0; i<2; ++i) BH[i] = *(++rand_f_it);
  BO[0] = *(++rand_f_it);

  //learn
  for (int i = 0; i<100000; ++i){
    //print current state
    float agg_loss = 0; //aggregate loss
    float outputs [4]; //to print later
    for(int j = 0; j<4; ++j){
      I[0] = BATCH[j][0]; I[1] = BATCH[j][1];
      forward();
      loss(BATCH[j][2]);
      agg_loss += L[0];
      outputs[j] = OS[0];
    }
    printf("EXPECTED: {%f, %f, %f, %f}, PREDICTED: {%f, %f, %f, %f}, LOSS: %.10f\n",
        BATCH[0][2], BATCH[1][2], BATCH[2][2], BATCH[3][2],
        outputs[0], outputs[1], outputs[2], outputs[3], agg_loss/4);


    //run one epoch
    learn(BATCH, 0.1);
  }

}

void forward(){
  H[0] = WH[0] * I[0] + WH[1] * I[1] + BH[0];
  HS[0] = softsign(H[0]);
  H[1] = WH[3] * I[1] + WH[2] * I[0] + BH[1];
  HS[1] = softsign(H[1]);
  O[0] = WO[0] * HS[0] + WO[1] * HS[1] + BO[0];
  OS[0] = softsign(O[0]);
}

void loss(float x){
  L[0] = (OS[0] -x)*(OS[0] -x);
}

void learn(const float batch[4][3], float rate){
  float GBO [1] = {0}; //aggregate grad of output bias
  float GWO [2] = {0}; //aggregate grad of output layer weights
  float GBH [2] = {0}; //aggregate grad of hidden layer biases
  float GWH [4] = {0}; //aggregate grad of hidden layer weights

  //run gradient calculations (hardcoded equations, derived by hand)
  for (int i = 0; i<4; ++i){
    I[0] = batch[i][0]; I[1] = batch[i][1];
    forward();
    loss(batch[i][2]);

    GBO[0] += 2*(OS[0] - batch[i][2])*softsign_der(O[0]);
    for(int j = 0; j<2; ++j)
      GWO[j] += HS[j] * 2*(OS[0] - batch[i][2])*softsign_der(O[0]);
    for(int j = 0; j<2; ++j)
      GBH[j] += softsign_der(H[j]) * WO[j] * 2*(OS[0] - batch[i][2])*softsign_der(O[0]);
    for(int j = 0; j<2; ++j)
      GWH[j] +=  softsign_der(H[0]) * WO[0] * 2*(OS[0] - batch[i][2])*softsign_der(O[0]) * I[j];
    for(int j = 0; j<2; ++j)
      GWH[j+2] += softsign_der(H[1]) * WO[1] * 2*(OS[0] - batch[i][2])*softsign_der(O[0]) * I[j];
  }

  //update weights and biases
  BO[0] -= rate * GBO[0] / 4;
  for(int j = 0; j<2; ++j)
    WO[j] -= rate * GWO[j] / 4;
  for(int j = 0; j<2; ++j)
    BH[j] -= rate * GBH[j] / 4;
  for(int j = 0; j<4; ++j)
    WH[j] -= rate * GWH[j] /4;

}

void U (unsigned int seed, float *u){
  for (int i = 0; i < 10; ++i){
    //simple RNG for range -1.4 to 1.4
    seed = seed ^ (seed >> 12); seed = seed ^ (seed >> 25); seed = seed ^ (seed >> 27);
    seed = seed * 0x2545F4914F6CDD1D;

    u[i] = (float)(seed % 10000) / 3500 - 1.4;
  }
}
