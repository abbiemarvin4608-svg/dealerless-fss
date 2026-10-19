#include <iostream>
#include <cstring>
#include <cmath>

#include "defines.h"
#include "datatypes.h"
#include "predictors.h"
#include "library_fixed.h"
#include "model_fixed.h"
#include "vars_fixed.h"

using namespace std;
using namespace sirnn_fixed;

extern int party;
extern string address;
extern int port;
extern int num_threads;
void sirnnFixed(MYINT *X_temp, int64_t* res) {
	initialize();
	
	int32_t __tac_var607 = ( (int32_t)32 *  (int32_t)16);
	
	
	
	
	uint64_t __tmp_in_W1_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var607; i0++){
		
		
	}
	
	int32_t __tac_var608 = ( (int32_t)16 *  (int32_t)100);
	
	
	
	
	uint64_t __tmp_in_W2_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var608; i0++){
		
		
	}
	
	int32_t __tac_var609 = ( (int32_t)100 *  (int32_t)35);
	
	
	
	
	uint64_t __tmp_in_U1_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var609; i0++){
		
		
	}
	
	int32_t __tac_var610 = ( (int32_t)35 *  (int32_t)100);
	
	
	
	
	uint64_t __tmp_in_U2_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var610; i0++){
		
		
	}
	
	int32_t __tac_var611 = ( (int32_t)1 *  (int32_t)100);
	
	
	
	
	uint64_t __tmp_in_Bg_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var611; i0++){
		
		
	}
	
	int32_t __tac_var612 = ( (int32_t)1 *  (int32_t)100);
	
	
	
	
	uint64_t __tmp_in_Bh_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var612; i0++){
		
		
	}
	
	int32_t __tac_var613 = ( (int32_t)100 *  (int32_t)30);
	
	
	
	
	uint64_t __tmp_in_FC_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var613; i0++){
		
		
	}
	
	int32_t __tac_var614 = ( (int32_t)1 *  (int32_t)30);
	
	
	
	
	uint64_t __tmp_in_FCbias_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var614; i0++){
		
		
	}
	
	int32_t __tac_var615 = ( (int32_t)3168 *  (int32_t)1);
	
	
	
	
	uint64_t __tmp_in_X_temp;
	for (uint32_t i0 =  (uint32_t)0; i0 < __tac_var615; i0++){
		
		
	}
	
	auto W1 = new int64_t[( (int32_t)32 *  (int32_t)16)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)32; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)16; i1++){
			
			int32_t __tac_var616 = (i0 *  (int32_t)16);
			
			int32_t __tac_var617 = (__tac_var616 + i1);
			
			int32_t __tac_var618 = (i0 *  (int32_t)16);
			
			int32_t __tac_var619 = (__tac_var618 + i1);
			
			int64_t __tac_var620 = W1_temp[__tac_var619];
			
			int64_t __tac_var621 =  (int64_t)256;
			W1[__tac_var617]  = ( party == 1 ?  (__tac_var620 / __tac_var621) : 0);
		}
	}
	
	auto W2 = new int64_t[( (int32_t)16 *  (int32_t)100)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)16; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)100; i1++){
			
			int32_t __tac_var622 = (i0 *  (int32_t)100);
			
			int32_t __tac_var623 = (__tac_var622 + i1);
			
			int32_t __tac_var624 = (i0 *  (int32_t)100);
			
			int32_t __tac_var625 = (__tac_var624 + i1);
			
			int64_t __tac_var626 = W2_temp[__tac_var625];
			
			int64_t __tac_var627 =  (int64_t)1;
			W2[__tac_var623]  = ( party == 1 ?  (__tac_var626 / __tac_var627) : 0);
		}
	}
	
	auto U1 = new int64_t[( (int32_t)100 *  (int32_t)35)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)100; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)35; i1++){
			
			int32_t __tac_var628 = (i0 *  (int32_t)35);
			
			int32_t __tac_var629 = (__tac_var628 + i1);
			
			int32_t __tac_var630 = (i0 *  (int32_t)35);
			
			int32_t __tac_var631 = (__tac_var630 + i1);
			
			int64_t __tac_var632 = U1_temp[__tac_var631];
			
			int64_t __tac_var633 =  (int64_t)256;
			U1[__tac_var629]  = ( party == 1 ?  (__tac_var632 / __tac_var633) : 0);
		}
	}
	
	auto U2 = new int64_t[( (int32_t)35 *  (int32_t)100)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)35; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)100; i1++){
			
			int32_t __tac_var634 = (i0 *  (int32_t)100);
			
			int32_t __tac_var635 = (__tac_var634 + i1);
			
			int32_t __tac_var636 = (i0 *  (int32_t)100);
			
			int32_t __tac_var637 = (__tac_var636 + i1);
			
			int64_t __tac_var638 = U2_temp[__tac_var637];
			
			int64_t __tac_var639 =  (int64_t)1;
			U2[__tac_var635]  = ( party == 1 ?  (__tac_var638 / __tac_var639) : 0);
		}
	}
	
	auto Bg = new int64_t[( (int32_t)1 *  (int32_t)100)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)1; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)100; i1++){
			
			int32_t __tac_var640 = (i0 *  (int32_t)100);
			
			int32_t __tac_var641 = (__tac_var640 + i1);
			
			int32_t __tac_var642 = (i0 *  (int32_t)100);
			
			int32_t __tac_var643 = (__tac_var642 + i1);
			
			int64_t __tac_var644 = Bg_temp[__tac_var643];
			
			int64_t __tac_var645 =  (int64_t)1;
			Bg[__tac_var641]  = ( party == 1 ?  (__tac_var644 / __tac_var645) : 0);
		}
	}
	
	auto Bh = new int64_t[( (int32_t)1 *  (int32_t)100)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)1; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)100; i1++){
			
			int32_t __tac_var646 = (i0 *  (int32_t)100);
			
			int32_t __tac_var647 = (__tac_var646 + i1);
			
			int32_t __tac_var648 = (i0 *  (int32_t)100);
			
			int32_t __tac_var649 = (__tac_var648 + i1);
			
			int64_t __tac_var650 = Bh_temp[__tac_var649];
			
			int64_t __tac_var651 =  (int64_t)1;
			Bh[__tac_var647]  = ( party == 1 ?  (__tac_var650 / __tac_var651) : 0);
		}
	}
	
	auto FC = new int64_t[( (int32_t)100 *  (int32_t)30)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)100; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)30; i1++){
			
			int32_t __tac_var652 = (i0 *  (int32_t)30);
			
			int32_t __tac_var653 = (__tac_var652 + i1);
			
			int32_t __tac_var654 = (i0 *  (int32_t)30);
			
			int32_t __tac_var655 = (__tac_var654 + i1);
			
			int64_t __tac_var656 = FC_temp[__tac_var655];
			
			int64_t __tac_var657 =  (int64_t)256;
			FC[__tac_var653]  = ( party == 1 ?  (__tac_var656 / __tac_var657) : 0);
		}
	}
	
	auto FCbias = new int64_t[( (int32_t)1 *  (int32_t)30)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)1; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)30; i1++){
			
			int32_t __tac_var658 = (i0 *  (int32_t)30);
			
			int32_t __tac_var659 = (__tac_var658 + i1);
			
			int32_t __tac_var660 = (i0 *  (int32_t)30);
			
			int32_t __tac_var661 = (__tac_var660 + i1);
			
			int64_t __tac_var662 = FCbias_temp[__tac_var661];
			
			int64_t __tac_var663 =  (int64_t)256;
			FCbias[__tac_var659]  = ( party == 1 ?  (__tac_var662 / __tac_var663) : 0);
		}
	}
	
	auto X = new int64_t[( (int32_t)3168 *  (int32_t)1)];
	for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)3168; i0++){
		for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)1; i1++){
			
			int32_t __tac_var664 = (i0 *  (int32_t)1);
			
			int32_t __tac_var665 = (__tac_var664 + i1);
			
			int32_t __tac_var666 = (i0 *  (int32_t)1);
			
			int32_t __tac_var667 = (__tac_var666 + i1);
			
			int64_t __tac_var668 = X_temp[__tac_var667];
			
			int64_t __tac_var669 =  (int64_t)1;
			X[__tac_var665]  = ( party == 2 ?  (__tac_var668 / __tac_var669) : 0);
		}
	}
	
	int64_t tmp9bw16;
	tmp9bw16 =  (int64_t)0;
	
	int64_t tmp10bw16;
	tmp10bw16 =  (int64_t)0;
	
	auto tmp11bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var670 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp11bw16 =  (int32_t)0; inittmp11bw16 < __tac_var670; inittmp11bw16++){
		tmp11bw16[inittmp11bw16] =  (int64_t)0;
	}
	
	auto tmp12bw16 = new int64_t[( (int32_t)32 *  (int32_t)1)];
	
	int32_t __tac_var671 = ( (int32_t)32 *  (int32_t)1);
	for (uint32_t inittmp12bw16 =  (int32_t)0; inittmp12bw16 < __tac_var671; inittmp12bw16++){
		tmp12bw16[inittmp12bw16] =  (int64_t)0;
	}
	
	int64_t tmp13;
	
	int64_t tmp14;
	
	auto tmp15bw16 = new int64_t[( (int32_t)1 *  (int32_t)32)];
	
	int32_t __tac_var672 = ( (int32_t)1 *  (int32_t)32);
	for (uint32_t inittmp15bw16 =  (int32_t)0; inittmp15bw16 < __tac_var672; inittmp15bw16++){
		tmp15bw16[inittmp15bw16] =  (int64_t)0;
	}
	
	int64_t tmp16;
	
	int64_t tmp17;
	
	auto tmp19bw16 = new int64_t[( (int32_t)1 *  (int32_t)16)];
	
	int32_t __tac_var673 = ( (int32_t)1 *  (int32_t)16);
	for (uint32_t inittmp19bw16 =  (int32_t)0; inittmp19bw16 < __tac_var673; inittmp19bw16++){
		tmp19bw16[inittmp19bw16] =  (int64_t)0;
	}
	
	auto tmp18bw32 = new int64_t[ (int32_t)32];
	for (uint32_t inittmp18bw32 =  (int32_t)0; inittmp18bw32 <  (int32_t)32; inittmp18bw32++){
		tmp18bw32[inittmp18bw32] =  (int64_t)0;
	}
	
	auto tmp21bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var674 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp21bw16 =  (int32_t)0; inittmp21bw16 < __tac_var674; inittmp21bw16++){
		tmp21bw16[inittmp21bw16] =  (int64_t)0;
	}
	
	auto tmp20bw32 = new int64_t[ (int32_t)16];
	for (uint32_t inittmp20bw32 =  (int32_t)0; inittmp20bw32 <  (int32_t)16; inittmp20bw32++){
		tmp20bw32[inittmp20bw32] =  (int64_t)0;
	}
	
	auto tmp23bw8 = new int64_t[( (int32_t)1 *  (int32_t)35)];
	
	int32_t __tac_var675 = ( (int32_t)1 *  (int32_t)35);
	for (uint32_t inittmp23bw8 =  (int32_t)0; inittmp23bw8 < __tac_var675; inittmp23bw8++){
		tmp23bw8[inittmp23bw8] =  (int64_t)0;
	}
	
	auto tmp22bw32 = new int64_t[ (int32_t)100];
	for (uint32_t inittmp22bw32 =  (int32_t)0; inittmp22bw32 <  (int32_t)100; inittmp22bw32++){
		tmp22bw32[inittmp22bw32] =  (int64_t)0;
	}
	
	auto tmp25bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var676 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp25bw16 =  (int32_t)0; inittmp25bw16 < __tac_var676; inittmp25bw16++){
		tmp25bw16[inittmp25bw16] =  (int64_t)0;
	}
	
	auto tmp24bw32 = new int64_t[ (int32_t)35];
	for (uint32_t inittmp24bw32 =  (int32_t)0; inittmp24bw32 <  (int32_t)35; inittmp24bw32++){
		tmp24bw32[inittmp24bw32] =  (int64_t)0;
	}
	
	auto tmp26bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var677 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp26bw16 =  (int32_t)0; inittmp26bw16 < __tac_var677; inittmp26bw16++){
		tmp26bw16[inittmp26bw16] =  (int64_t)0;
	}
	
	auto tmp27bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var678 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp27bw16 =  (int32_t)0; inittmp27bw16 < __tac_var678; inittmp27bw16++){
		tmp27bw16[inittmp27bw16] =  (int64_t)0;
	}
	
	auto tmp28bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var679 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp28bw16 =  (int32_t)0; inittmp28bw16 < __tac_var679; inittmp28bw16++){
		tmp28bw16[inittmp28bw16] =  (int64_t)0;
	}
	
	auto tmp29bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var680 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp29bw16 =  (int32_t)0; inittmp29bw16 < __tac_var680; inittmp29bw16++){
		tmp29bw16[inittmp29bw16] =  (int64_t)0;
	}
	
	auto tmp30bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var681 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp30bw16 =  (int32_t)0; inittmp30bw16 < __tac_var681; inittmp30bw16++){
		tmp30bw16[inittmp30bw16] =  (int64_t)0;
	}
	
	auto tmp31bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var682 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp31bw16 =  (int32_t)0; inittmp31bw16 < __tac_var682; inittmp31bw16++){
		tmp31bw16[inittmp31bw16] =  (int64_t)0;
	}
	
	int64_t tmp32;
	tmp32 =  (int64_t)0;
	
	auto tmp33bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var683 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp33bw16 =  (int32_t)0; inittmp33bw16 < __tac_var683; inittmp33bw16++){
		tmp33bw16[inittmp33bw16] =  (int64_t)0;
	}
	
	auto tmp34bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var684 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp34bw16 =  (int32_t)0; inittmp34bw16 < __tac_var684; inittmp34bw16++){
		tmp34bw16[inittmp34bw16] =  (int64_t)0;
	}
	
	auto tmp35bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var685 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp35bw16 =  (int32_t)0; inittmp35bw16 < __tac_var685; inittmp35bw16++){
		tmp35bw16[inittmp35bw16] =  (int64_t)0;
	}
	
	auto tmp36bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var686 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp36bw16 =  (int32_t)0; inittmp36bw16 < __tac_var686; inittmp36bw16++){
		tmp36bw16[inittmp36bw16] =  (int64_t)0;
	}
	
	auto tmp37bw16 = new int64_t[( (int32_t)1 *  (int32_t)100)];
	
	int32_t __tac_var687 = ( (int32_t)1 *  (int32_t)100);
	for (uint32_t inittmp37bw16 =  (int32_t)0; inittmp37bw16 < __tac_var687; inittmp37bw16++){
		tmp37bw16[inittmp37bw16] =  (int64_t)0;
	}
	
	auto tmp39bw8 = new int64_t[( (int32_t)1 *  (int32_t)30)];
	
	int32_t __tac_var688 = ( (int32_t)1 *  (int32_t)30);
	for (uint32_t inittmp39bw8 =  (int32_t)0; inittmp39bw8 < __tac_var688; inittmp39bw8++){
		tmp39bw8[inittmp39bw8] =  (int64_t)0;
	}
	
	auto tmp38bw32 = new int64_t[ (int32_t)100];
	for (uint32_t inittmp38bw32 =  (int32_t)0; inittmp38bw32 <  (int32_t)100; inittmp38bw32++){
		tmp38bw32[inittmp38bw32] =  (int64_t)0;
	}
	
	auto tmp40bw8 = new int64_t[( (int32_t)1 *  (int32_t)30)];
	
	int32_t __tac_var689 = ( (int32_t)1 *  (int32_t)30);
	for (uint32_t inittmp40bw8 =  (int32_t)0; inittmp40bw8 < __tac_var689; inittmp40bw8++){
		tmp40bw8[inittmp40bw8] =  (int64_t)0;
	}
	
	auto tmp41 = new int64_t[ (int32_t)1];
	tmp9bw16 =  (int64_t)16654;
	tmp10bw16 =  (int64_t)26006;
	tmp32 =  (int64_t)16384;
	for (uint32_t msetVartmp11 =  (int32_t)0; msetVartmp11 <  (int32_t)100; msetVartmp11++){
		tmp11bw16[msetVartmp11] =  (int64_t)0;
	}
	for (uint32_t i =  (int32_t)0; i <  (int32_t)99; i++){
		for (uint32_t i0 =  (int32_t)0; i0 <  (int32_t)32; i0++){
			for (uint32_t i1 =  (int32_t)0; i1 <  (int32_t)1; i1++){
				
				int64_t __tac_var690 = ( (int64_t)32 * i);
				tmp13 = (i0 + __tac_var690);
				tmp14 = (i1 +  (int64_t)0);
				
				int32_t __tac_var691 = (i0 *  (int32_t)1);
				
				int32_t __tac_var692 = (__tac_var691 + i1);
				
				int64_t __tac_var693 = (tmp13 *  (int32_t)1);
				
				int64_t __tac_var694 = (__tac_var693 + tmp14);
				tmp12bw16[__tac_var692] = X[__tac_var694];
			}
		}
		for (uint32_t mcpyVartmp12bw16tmp15bw16 =  (int32_t)0; mcpyVartmp12bw16tmp15bw16 <  (int32_t)32; mcpyVartmp12bw16tmp15bw16++){
			
			int32_t __tac_var695 = ( (int32_t)0 *  (int32_t)32);
			
			int32_t __tac_var696 = (__tac_var695 +  (int32_t)0);
			
			int32_t __tac_var697 = (__tac_var696 + mcpyVartmp12bw16tmp15bw16);
			
			int32_t __tac_var698 = ( (int32_t)0 *  (int32_t)1);
			
			int32_t __tac_var699 = (__tac_var698 +  (int32_t)0);
			
			int32_t __tac_var700 = (__tac_var699 + mcpyVartmp12bw16tmp15bw16);
			tmp15bw16[__tac_var697] = tmp12bw16[__tac_var700];
		}
		MatMul( (int32_t)1,  (int32_t)32,  (int32_t)16,  (int32_t)1,  (int32_t)2,  (int32_t)5,  (int32_t)0,  (int32_t)1,  (int32_t)16,  (int32_t)8,  (int32_t)32,  (int32_t)16, tmp15bw16, W1, tmp19bw16, tmp18bw32);
		MatMul( (int32_t)1,  (int32_t)16,  (int32_t)100,  (int32_t)32,  (int32_t)64,  (int32_t)4,  (int32_t)0,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)32,  (int32_t)16, tmp19bw16, W2, tmp21bw16, tmp20bw32);
		MatMul( (int32_t)1,  (int32_t)100,  (int32_t)35,  (int32_t)1,  (int32_t)1,  (int32_t)6,  (int32_t)1,  (int32_t)256,  (int32_t)16,  (int32_t)8,  (int32_t)32,  (int32_t)8, tmp11bw16, U1, tmp23bw8, tmp22bw32);
		MatMul( (int32_t)1,  (int32_t)35,  (int32_t)100,  (int32_t)1,  (int32_t)1,  (int32_t)6,  (int32_t)0,  (int32_t)1,  (int32_t)8,  (int32_t)16,  (int32_t)32,  (int32_t)16, tmp23bw8, U2, tmp25bw16, tmp24bw32);
		MatAdd( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)8,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp21bw16, tmp25bw16, tmp26bw16);
		MatAdd( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)8,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp26bw16, Bg, tmp27bw16);
		Sigmoid( (int32_t)1,  (int32_t)100,  (int32_t)128,  (int32_t)16384,  (int32_t)16,  (int32_t)16, tmp27bw16, tmp28bw16);
		MatAdd( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)8,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp26bw16, Bh, tmp29bw16);
		TanH( (int32_t)1,  (int32_t)100,  (int32_t)128,  (int32_t)128,  (int32_t)16,  (int32_t)16, tmp29bw16, tmp30bw16);
		MulCir( (int32_t)1,  (int32_t)100,  (int32_t)128,  (int32_t)128,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)32,  (int32_t)16, tmp28bw16, tmp11bw16, tmp31bw16);
		MatSubBroadCastA( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp32, tmp28bw16, tmp33bw16);
		ScalarMul( (int32_t)1,  (int32_t)100,  (int32_t)128,  (int32_t)128,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)32,  (int32_t)16, tmp9bw16, tmp33bw16, tmp34bw16);
		MatAddBroadCastB( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp34bw16, tmp10bw16, tmp35bw16);
		MulCir( (int32_t)1,  (int32_t)100,  (int32_t)8,  (int32_t)16,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)32,  (int32_t)16, tmp35bw16, tmp30bw16, tmp36bw16);
		MatAdd( (int32_t)1,  (int32_t)100,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)16,  (int32_t)16, tmp31bw16, tmp36bw16, tmp11bw16);
	}
	MatMul( (int32_t)1,  (int32_t)100,  (int32_t)30,  (int32_t)1,  (int32_t)2,  (int32_t)7,  (int32_t)0,  (int32_t)256,  (int32_t)16,  (int32_t)8,  (int32_t)32,  (int32_t)8, tmp11bw16, FC, tmp39bw8, tmp38bw32);
	MatAdd( (int32_t)1,  (int32_t)30,  (int32_t)1,  (int32_t)8,  (int32_t)1,  (int32_t)1,  (int32_t)8,  (int32_t)8,  (int32_t)8,  (int32_t)8, tmp39bw8, FCbias, tmp40bw8);
	int64_t __sirnn_argmax_input_16[30];
	xt->s_extend((int32_t)30, (uint64_t*)tmp40bw8, (uint64_t*)__sirnn_argmax_input_16, (int32_t)8, (int32_t)16);
	ArgMax( (int32_t)1,  (int32_t)30,  (int32_t)16,  (int32_t)8, __sirnn_argmax_input_16, tmp41);
	
	reconstruct(tmp41, res, 1, ( (int32_t)1), 8);
	
	delete[] W1;
	delete[] W2;
	delete[] U1;
	delete[] U2;
	delete[] Bg;
	delete[] Bh;
	delete[] FC;
	delete[] FCbias;
	delete[] X;
	delete[] tmp11bw16;
	delete[] tmp12bw16;
	delete[] tmp15bw16;
	delete[] tmp19bw16;
	delete[] tmp18bw32;
	delete[] tmp21bw16;
	delete[] tmp20bw32;
	delete[] tmp23bw8;
	delete[] tmp22bw32;
	delete[] tmp25bw16;
	delete[] tmp24bw32;
	delete[] tmp26bw16;
	delete[] tmp27bw16;
	delete[] tmp28bw16;
	delete[] tmp29bw16;
	delete[] tmp30bw16;
	delete[] tmp31bw16;
	delete[] tmp33bw16;
	delete[] tmp34bw16;
	delete[] tmp35bw16;
	delete[] tmp36bw16;
	delete[] tmp37bw16;
	delete[] tmp39bw8;
	delete[] tmp38bw32;
	delete[] tmp40bw8;
	delete[] tmp41;
	finalize();
	return;
}

const int switches = 0;
void sirnnFixedSwitch(int i, MYINT **X_temp, int32_t* res) {
		switch(i) {
				default: res[0] = -1; return;
		}
}
