Class CFdmtC
1. How to create instance:
CFdmtC *pfdmt = new CFdmtC(
	   Fmin
	,  Fmax	
	,  nchan // quant channels/rows of input image
	,  cols		
        ,  imaxDt // quantity of rows of output image
        );
2. How to call calculations:
pfdmt->process_image(piarrImage, piarrImOut, false);

here:
input:
piarrImage - one-dimentional array type of fdmt_type_, length = nchan*cols

output:
piarrImOut - one-dimentional array type of fdmt_type_, imaxDt = nchan*cols

fdmt_type_ defines in Constants.h

3. If you need to calculate nomalizing array applay the following call:
pfdmt->process_image(NULL, piarrImOut, true);  

