clear all
close all
clc

%%��������ͼ
figure(1);
lbx=-2;ubx=2;%�����Ա�����Χ[-2,2]
lby=-2;uby=2;
ezmesh('x*cos(2*pi*y)+y*sin(2*pi*x)',[lbx,ubx,lby,uby],50);%������������,�з�Χ�Ļ�ͼ��������
hold on;

%%�����Ŵ��㷨����
NIND=40;%��Ⱥ��С
MAXGEN=50;%����Ŵ�����
PRECI=20;%���峤��
GGAP=0.95;%����
px=0.7;%�������
pm=0.01;%�������
trace=zeros(3,MAXGEN);%Ѱ�Ž���ĳ�ʼֵ
FieldD=[PRECI PRECI;lbx lby;ubx uby;1 1;0 0;1 1;1 1];%����������
Chrom=crtbp(NIND,PRECI*2);%����������ɢ�����Ⱥ

%%�Ż�
gen=0;%��������
XY=bs2rv(Chrom,FieldD);%��ʼ����Ⱥ�����Ƶ�ʮ����ת��
X=XY(:,1);Y=XY(:,2);
ObjV=Y.*sin(2*pi*X)+X.*cos(2*pi*Y);%����Ŀ�꺯��ֵ
while gen<MAXGEN
    FitnV=ranking(-ObjV);%������Ӧ��ֵ,�����ֵҪ�Ӹ���
    SelCh=select('sus',Chrom,FitnV,GGAP);%ѡ��
    SelCh=recombin('xovsp',SelCh,px);%����
    SelCh=mut(SelCh,pm);%����
    XY=bs2rv(SelCh,FieldD);%�Ӵ������ʮ����ת��
    X=XY(:,1);Y=XY(:,2);
    ObjVSel=Y.*sin(2*pi*X)+X.*cos(2*pi*Y);%�����Ӵ���Ŀ�꺯��ֵ
    [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);%�ز����Ӵ����������õ�����Ⱥ
    XY=bs2rv(SelCh,FieldD);
    gen=gen+1;%������������
    %��ȡÿ�������Ž⼰����ţ�YΪ���Ž⣬IΪ��������
    [Y,I]=max(ObjV);
    trace(1:2,gen)=XY(I,:);
    trace(3,gen)=Y;%����ÿ��������ֵ
end
plot3(trace(1,:),trace(2,:),trace(3,:),'bo');%����ÿ��������ֵ
grid on;
plot3(XY(:,1),XY(:,2),ObjV,'b *');%�������һ������Ⱥ
hold off

%������ͼ
figure(2);
plot(1:MAXGEN,trace(3,:));
grid on;
xlabel('�Ŵ�����')
ylabel('��ı仯')
title('��������')
bestZ=trace(3,end);
bestY=trace(2,end);
bestX=trace(1,end);
fprintf(['���Ž⣺\nX=',num2str(bestX),'\nY=',num2str(bestY),'\nZ=',num2str(bestZ),'\n'])