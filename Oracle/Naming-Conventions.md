https://trivadis.github.io/plsql-and-sql-coding-guidelines/v4.0/2-naming-conventions/naming-conventions/
### Naming Conventions  
```
Identifier	              Prefix	   Suffix	    Example
-----------------------   ---------  ---------  --------------------
Global Variable	          g_		                g_version
Local Variable	          l_		  	      	    l_version
Cursor	                  c_			      	      c_employees
Record	                  r_			         	    r_employee
Array / Table	            t_		 	      	      t_employees
-----------------------   ---------  ---------  --------------------
Object	                  o_			    	        o_employee
Cursor Parameter	        p_	    	    		    p_empno
In Parameter	            in_	    	       	    in_empno
Out Parameter		          out_	    	      		out_ename
In/Out Parameter	        io_			              io_employee
-----------------------   ---------  ---------  --------------------
Record Type Definitions	  r_	       _type	    r_employee_type
Array/Table Type          t_         _type	    t_employees_type
Exception	                e_                    e_employee_exists
Constants	                co_	                  co_empno
Subtypes		                         _type      big_string_type
-----------------------   ---------  ---------  --------------------
```
