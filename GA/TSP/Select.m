function SelCh = Select( Chrom,FitnV,GGAP )
%%ѡ�����
%���룺
%Chrom  ��Ⱥ
%FitnV  ��Ӧ��ֵ
%GGAP   ѡ�����
%�����
%SelCh  ��ѡ��ĸ���

NIND=size(Chrom,1);%��Ⱥ��С
NSel=max(floor(NIND*GGAP+.5),2);
ChrIx=Sus(FitnV,NSel);
SelCh=Chrom(ChrIx,:);

end

