package at.rovo.classifier.svm;

import at.rovo.classifier.svm.utils.Utils;

/**
 * A cache for at least <em>totalDataItems</em> elements, which are stored in a linked list. Elements are inserted at
 * the end of the cache and removed in first-in first-out behavior. On accessing a element in the cache, it is moved to
 * the end of the cache.
 *
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Cache
{
	/** The available cache size limit in bytes **/
	private long size;

	private final class head_t
	{
		head_t prev, next; // a cicular list
		float[] data;
		int len; // data[0,len) is cached in this entry
	}

	/** Stores the actual data **/
	private final head_t[] head;
	/** Pointer to the head of the list **/
	private head_t lruHead;

	/**
	 * Initializes a new instance of the cache.
	 *
	 * @param totalDataItems
	 * 		The number of data items to be conducted by the cache
	 * @param size
	 * 		The size in bytes available to the cache
	 */
	public Cache(int totalDataItems, long size)
	{
		this.size = size;
		this.head = new head_t[totalDataItems];
		for (int i = 0; i < totalDataItems; i++)
		{
			this.head[i] = new head_t();
		}
		this.size /= 4;
		this.size -= totalDataItems * (16 / 4); // sizeof(head_t) == 16
		// cache must be large enough for two columns
		this.size = Math.max(this.size, 2 * (long) totalDataItems);

		this.lruHead = new head_t();
		this.lruHead.next = this.lruHead.prev = this.lruHead;
	}

	/**
	 * Deletes an element inside a linked list. It sets the pointers to the successor and predecessor node accordingly.
	 *
	 * @param h
	 * 		The element to delete from the linked list
	 */
	private void lruDelete(head_t h)
	{
		// delete from current location
		h.prev.next = h.next;
		h.next.prev = h.prev;
	}

	/**
	 * Inserts a element into a linked list at the end of the list.
	 *
	 * @param h
	 * 		The element to be inserted at the end of the linked list
	 */
	private void lruInsert(head_t h)
	{
		// insert to last position
		h.next = this.lruHead;
		h.prev = this.lruHead.prev;
		h.prev.next = h;
		h.next.prev = h;
	}

	/**
	 * Checks if enough cache space is available for the data to be stored else the cache is cleaned from old data until
	 * the new data will fit in. Data is deleted in first-in-first-out order.
	 *
	 * @param index
	 * 		The index to add the data into the cache
	 * @param data
	 * 		The data to add to the cache
	 * @param len
	 * 		The required length in bytes to add to the cache
	 *
	 * @return Some position p where [p,len) need to be filled
	 */
	public int getData(int index, float[][] data, int len)
	{
		head_t h = this.head[index];
		if (h.len > 0)
		{
			lruDelete(h);
		}
		int more = len - h.len;

		// more space is required, so we need to free some old space
		if (more > 0)
		{
			// free old space
			while (this.size < more)
			{
				head_t old = this.lruHead.next;
				lruDelete(old);
				this.size += old.len;
				old.data = null;
				old.len = 0;
			}

			// allocate new space and set the data for this element
			float[] new_data = new float[len];
			if (h.data != null)
			{
				System.arraycopy(h.data, 0, new_data, 0, h.len);
			}
			h.data = new_data;
			this.size -= more;
			// swap the length
			int _i = h.len;
			h.len = len;
			len = _i;
		}

		// enough space is available, so add the element with the data to the
		// end of the cache
		lruInsert(h);
		data[0] = h.data;
		return len;
	}

	/**
	 * Swaps the index of element i in the cache with element j in the cache and vice versa.
	 *
	 * @param i
	 * 		Element i in the cache
	 * @param j
	 * 		Element j in the cache
	 */
	public void swapIndex(int i, int j)
	{
		if (i == j)
		{
			return;
		}

		// delete the elements at position i and j if they are not empty
		if (this.head[i].len > 0)
		{
			lruDelete(this.head[i]);
		}
		if (this.head[j].len > 0)
		{
			lruDelete(this.head[j]);
		}

		// swap the data fields of elements i and j
		float[] _f = this.head[i].data;
		this.head[i].data = this.head[j].data;
		this.head[j].data = _f;

		// swap the length fields of element i and j
		int _i = this.head[i].len;
		this.head[i].len = this.head[j].len;
		this.head[j].len = _i;

		// insert the elements at the end of the list
		if (this.head[i].len > 0)
		{
			lruInsert(this.head[i]);
		}
		if (this.head[j].len > 0)
		{
			lruInsert(this.head[j]);
		}

		// swap i with j if i is greater than j
		if (i > j)
		{
			_i = i;
			i = j;
			j = _i;
		}

		// runs through every element in the linked list and swaps the data
		// between the i'th and j'th field. If an element has at least i data
		// fields but does not have at least j data-fields it is removed from 
		// the element is removed from the cache and the space is freed 
		for (head_t h = this.lruHead.next; h != this.lruHead; h = h.next)
		{
			if (h.len > i)
			{
				if (h.len > j)
				{
					Utils.swap(h.data, i, j);
				}
				else
				{
					// give up
					lruDelete(h);
					this.size += h.len;
					h.data = null;
					h.len = 0;
				}
			}
		}
	}
}